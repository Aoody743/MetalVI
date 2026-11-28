#!/usr/bin/env python3
"""T1 standard run report: one Markdown table over check-intro.sh run directories.

    metal_run_report.py RUN_DIR [RUN_DIR ...] [--3d-frame N] [--png-limit N]

Each RUN_DIR is a check-intro.sh output directory (game.log, frames/,
metal-cache/, system-gpu.log, hang-sample-*.txt, window/). Siblings
RUN_DIR.supervisor.log (check-intro's stderr) and RUN_DIR.env (what
run-report.sh ran it with) are read when present. Columns are runs, rows
are the checks of plan v3 section 5 (V0 and T1). Details (host-phase
hangs, crash lines, capture PNG paths) follow the table.

Reaching 3D: the first indirect execution of the title's world rendering
("indirect execution dropped (first time ...)" on the D3D12 side, or
"Metal: indirect arguments clamped" / "indirect record clamped" on the
Metal side). Every run of 2026-11-27/28 that reached the first 3D dialogue
shows it at frame ~900-1300, and runs stuck in the loading loop never do.
--3d-frame N takes the first frame >= N instead.

Frame times: the T8 line carries the 64 present-to-present times (`ms=`)
since v3-tools; percentiles and stalls are then exact. Older logs only
have per-64-frame p50/p95/max, so the report shows "≈" values: the median
of the window p50s, the median of the window p95s (in the p90 column) and
the largest max (in the p99 column), and stalls as "≥N" (windows whose max
was over 1 s).
"""

import argparse
import datetime as dt
import json
import os
import re
import statistics
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z\s+(\w+)\s+(\S+):\s(.*)$")
FIELDS = re.compile(r'\b(\w+)=("[^"]*"|\[[^\]]*\]|Some\([^)]*\)|[^ ]+)')
FRAME = re.compile(r"\bframe=(\d+)\b")

THREE_D_MARKERS = (
    "indirect execution dropped (first time",
    "Metal: indirect arguments clamped",
    "Metal: indirect record clamped",
)


def fields(message):
    out = {}
    for key, value in FIELDS.findall(message):
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("Some("):
            value = value[5:-1]
        out[key] = value
    return out


def num(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


def pct(sorted_values, p):
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int((len(sorted_values) - 1) * p))]


class Run:
    def __init__(self, path, three_d_frame=None):
        self.path = Path(path).resolve()
        self.name = self.path.name
        self.three_d_frame = three_d_frame
        self.t0 = self.t_end = None
        self.frames = []          # (t, frame) in log order
        self.windows = []         # (t, frame, p50, p95, max, [ms] or None)
        self.three_d = None       # (t, frame, marker)
        self.counts = dict(fuse=0, fuse_lines=[], clamp_lines=0, clamp_records=0,
                           record_clamped=0, loop_cut=0, loop_threads=0, loop_waves=0,
                           half_sync=0, gpu_hang=0, draws_stop=0, no_metal_pipeline=0,
                           shader_refused=0)
        self.stuck = []
        self.crash = []
        self.shader_cache = None
        self.archive = None
        self.span = None
        self.sections = None
        self.prewire = dict(wired=0, failed=0, capped=0, freed=0, prewired_mb=0)
        self.busy = []
        self.draw_outcomes = None
        self.last_presented = None
        self.parse_log()
        self.supervisor = self.read_text(Path(str(self.path) + ".supervisor.log")).strip()
        self.env = self.read_text(Path(str(self.path) + ".env")).strip()
        self.system_gpu = [l for l in self.read_text(self.path / "system-gpu.log").splitlines()
                           if l[:4].isdigit() and l[4:5] == "-"]
        self.hang_samples = sorted(str(p) for p in self.path.glob("hang-sample-*.txt"))

    @staticmethod
    def read_text(path):
        try:
            return path.read_text(errors="replace")
        except OSError:
            return ""

    def time(self, stamp):
        t = dt.datetime.strptime(stamp[:26], "%Y-%m-%dT%H:%M:%S.%f").timestamp()
        if self.t0 is None:
            self.t0 = t
        self.t_end = t
        return t - self.t0

    def frame_at(self):
        return self.frames[-1][1] if self.frames else 0

    def parse_log(self):
        log = self.path / "game.log"
        if not log.exists():
            self.crash.append("no game.log")
            return
        t = 0.0
        with open(log, errors="replace") as f:
            for raw in f:
                line = ANSI.sub("", raw.rstrip("\n"))
                m = STAMP.match(line)
                if not m:
                    self.plain_line(line, t)
                    continue
                t = self.time(m.group(1))
                target, msg = m.group(3), m.group(4)
                self.message(t, target, msg, line)

    def plain_line(self, line, t):
        # Untimestamped lines: Metal validation aborts, MSC aborts, exceptions.
        keys = ("failed assertion", "unhandled exception", "Fatal:", "panicked at",
                "greater than the maximum", "not compatible with texture view")
        if any(k in line for k in keys):
            self.crash.append(f"+{t:.0f}s {line[:300]}")

    def note_frame(self, t, msg):
        m = FRAME.search(msg)
        if m:
            frame = int(m.group(1))
            if not self.frames or frame >= self.frames[-1][1]:
                self.frames.append((t, frame))
            return frame
        return None

    def message(self, t, target, msg, line):
        c = self.counts
        if target == "victory_gpu::queue":
            if msg.startswith("T8 present-to-present"):
                f = fields(msg)
                frame = self.note_frame(t, msg)
                ms = [int(x) for x in f["ms"].split(",") if x] if "ms" in f else None
                self.windows.append((t, frame, num(f.get("p50_ms")), num(f.get("p95_ms")),
                                     num(f.get("max_ms")), ms))
            elif msg.startswith("frame presented"):
                self.note_frame(t, msg)
                self.last_presented = fields(msg)
            elif msg.startswith("Metal capture: writes to the presented texture"):
                self.note_frame(t, msg)
            elif msg.startswith("shaders from the cache"):
                self.shader_cache = fields(msg)
        if self.three_d is None:
            for marker in THREE_D_MARKERS:
                if msg.startswith(marker):
                    self.three_d = (t, self.frame_at(), marker)
        if "Metal safety stop" in msg:
            c["fuse"] += 1
            if len(c["fuse_lines"]) < 3:
                c["fuse_lines"].append(f"+{t:.0f}s {msg[:300]}")
        elif msg.startswith("Metal: indirect arguments clamped"):
            c["clamp_lines"] += 1
            c["clamp_records"] += num(fields(msg).get("records"))
        elif msg.startswith("Metal: indirect record clamped"):
            c["record_clamped"] += 1
        elif msg.startswith("Metal: a loop budget cut shader threads"):
            c["loop_cut"] += 1
            c["loop_threads"] += num(fields(msg).get("threads"))
        elif msg.startswith("Metal: a loop budget stopped shader waves"):
            c["loop_cut"] += 1
            c["loop_waves"] += num(fields(msg).get("waves"))
        elif msg.startswith("Metal: half-wave sync timed out"):
            c["half_sync"] += 1
        elif "has not completed (GPU hang?)" in msg:
            c["gpu_hang"] += 1
            self.crash.append(f"+{t:.0f}s {msg[:300]}")
        elif msg.startswith("Metal: stuck in a host phase"):
            self.stuck.append(f"+{t:.0f}s frame≈{self.frame_at()} {msg}")
        elif msg.startswith("Metal: draws stop for this run"):
            c["draws_stop"] += 1
            self.crash.append(f"+{t:.0f}s {msg[:300]}")
        elif msg.startswith("no Metal pipeline for"):
            c["no_metal_pipeline"] += 1
        elif msg.startswith("shader does not translate") or "this shader is refused" in msg:
            c["shader_refused"] += 1
        elif msg.startswith("Metal: title memory span buffers"):
            self.span = fields(msg)
        elif msg.startswith("Metal: a persistent section view wired whole"):
            f = fields(msg)
            self.prewire["wired" if f.get("ok") == "true" else "failed"] += 1
            self.prewire["prewired_mb"] = num(f.get("prewired_mb"))
        elif msg.startswith("Metal: a persistent section view left to spans"):
            self.prewire["capped"] += 1
        elif msg.startswith("Metal: a section freed while its prewired buffer"):
            self.prewire["freed"] += 1
        elif msg.startswith("Metal: a persistent section view"):
            self.sections = fields(msg)
        elif msg.startswith("Metal: GPU busy share"):
            f = fields(msg)
            self.busy.append((num(f.get("wall_ms")), num(f.get("gpu_ms"))))
        elif msg.startswith("Metal draws: outcomes"):
            self.draw_outcomes = fields(msg)
        elif msg.startswith("Metal pipeline archive"):
            self.archive = fields(msg)
        elif "unhandled exception" in msg or "BacktraceBreakpad" in msg and "crashcontext" in msg:
            self.crash.append(f"+{t:.0f}s {msg[:300]}")

    # ---- derived values -------------------------------------------------

    def duration(self):
        return (self.t_end - self.t0) if self.t0 is not None else 0.0

    def start_3d(self):
        if self.three_d_frame is not None:
            for t, frame in self.frames:
                if frame >= self.three_d_frame:
                    return (t, frame, f"frame >= {self.three_d_frame}")
            return None
        return self.three_d

    def frame_times(self, from_frame):
        """Exact per-frame times after `from_frame`, or None for old logs."""
        times = []
        for _, frame, _, _, _, ms in self.windows:
            if ms is None:
                return None
            first = frame - len(ms) + 1
            times += [v for i, v in enumerate(ms) if first + i > from_frame]
        return times

    def frame_stats(self):
        s = self.start_3d()
        from_frame = s[1] if s else 0
        tail = 0.0
        if self.frames:
            tail = (self.t0 + self.duration()) - (self.t0 + self.frames[-1][0])
        times = self.frame_times(from_frame)
        if times is not None:
            if not times:
                return "-", "-", 0
            st = sorted(times)
            stalls = sum(1 for v in times if v > 1000) + (1 if tail > 1 else 0)
            longest = max(max(times), tail * 1000)
            return (f"{pct(st, .5)} / {pct(st, .9)} / {pct(st, .99)}",
                    f"{stalls} (最长 {longest / 1000:.1f} s)", len(times))
        win = [w for w in self.windows if w[1] is not None and w[1] - 63 > from_frame]
        if not win:
            return "-", (f"≥1 (末尾 {tail:.1f} s 无 present)" if tail > 1 else "-"), 0
        p50 = statistics.median(w[2] for w in win)
        p95 = statistics.median(w[3] for w in win)
        mx = max(w[4] for w in win)
        stalls = sum(1 for w in win if w[4] > 1000) + (1 if tail > 1 else 0)
        longest = max(mx, tail * 1000)
        return (f"≈{p50:.0f} / ≈{p95:.0f}(p95) / max {mx:.0f}",
                f"≥{stalls} (最长 {longest / 1000:.1f} s)", 64 * len(win))

    def fps_3d(self):
        s = self.start_3d()
        if not s or not self.frames:
            return None, None
        t3, f3, _ = s
        span = self.duration() - t3
        last = self.frames[-1][1]
        if span <= 0:
            return None, span
        return (last - f3) / span, span

    def metal_cache(self):
        d = self.path / "metal-cache"
        if not d.is_dir():
            return "-"
        try:
            start = (self.path / "frames").stat().st_birthtime
        except (OSError, AttributeError):
            start = self.path.stat().st_mtime
        old = new = 0
        with os.scandir(d) as it:
            for e in it:
                if e.stat().st_mtime >= start:
                    new += 1
                else:
                    old += 1
        return f"{old} 条预热 / {new} 条新编译"

    def captures(self):
        d = self.path / "frames"
        if not d.is_dir():
            return [], []
        frames = sorted(p for p in d.glob("frame*.png") if re.fullmatch(r"frame\d+\.png", p.name))
        finals = sorted(d.glob("final-draw*.png"))
        return frames, finals

    def end_reason(self):
        parts = []
        if self.supervisor:
            parts.append(self.supervisor.splitlines()[-1].replace("check-intro: ", ""))
        if self.system_gpu:
            parts.append(f"系统看门狗/GPU reset {len(self.system_gpu)} 行")
        if self.counts["fuse"]:
            parts.append("熔断")
        if self.crash:
            parts.append(self.crash[-1].split(" ", 1)[1][:80])
        if self.hang_samples:
            parts.append(f"{len(self.hang_samples)} 个 hang-sample")
        return "; ".join(parts) or "（无 supervisor 日志，进程自行退出？）"


def cell(v):
    return str(v).replace("|", "\\|").replace("\n", " ")


def report(runs, png_limit):
    rows = []

    def row(name, fn):
        rows.append([name] + [cell(fn(r)) for r in runs])

    row("环境", lambda r: r.env.replace("\n", " ") or "-")
    row("结束原因", Run.end_reason)
    row("日志时长 (s)", lambda r: f"{r.duration():.0f}")
    row("最高帧", lambda r: r.frame_at())

    def three(r):
        s = r.start_3d()
        return f"帧 {s[1]} @ +{s[0]:.0f}s" if s else "未进 3D"
    row("进 3D", three)

    def fps(r):
        v, span = r.fps_3d()
        return f"{v:.2f}（{span:.0f} s）" if v is not None else "-"
    row("3D 真实 fps（含停顿）", fps)
    stats = {r.name: r.frame_stats() for r in runs}
    row("3D 帧时间 p50/p90/p99 (ms)", lambda r: stats[r.name][0])
    row("3D 停顿 > 1 s", lambda r: stats[r.name][1])
    row("熔断", lambda r: f"触发（{r.counts['fuse']} 行）" if r.counts["fuse"] else "0")
    row("间接参数钳位（行/记录）", lambda r: f"{r.counts['clamp_lines']} / {r.counts['clamp_records']}"
        + (f" (+{r.counts['record_clamped']} trace)" if r.counts["record_clamped"] else ""))
    row("循环预算（行/线程/波）", lambda r: f"{r.counts['loop_cut']} / {r.counts['loop_threads']} / {r.counts['loop_waves']}")
    row("半波同步超时", lambda r: r.counts["half_sync"])
    row("GPU hang / 主机阶段卡住", lambda r: f"{r.counts['gpu_hang']} / {len(r.stuck)}")
    row("着色器缓存 hits/misses", lambda r: (f"{r.shader_cache.get('hits')} / {r.shader_cache.get('misses')}"
                                         if r.shader_cache else "-"))
    row("Metal 缓存（MSC）", Run.metal_cache)
    row("Metal 管线归档", lambda r: (" ".join(f"{k}={r.archive[k]}" for k in ("hits", "misses", "failed", "loaded")
                                         if k in r.archive) if r.archive else "-"))
    row("无 Metal 管线 / 翻译拒绝", lambda r: f"{r.counts['no_metal_pipeline']} / {r.counts['shader_refused']}")
    row("最后 present 行 draws/drawn/dispatches", lambda r: (
        f"{r.last_presented.get('frame')}: {r.last_presented.get('draws')}/{r.last_presented.get('drawn')}/"
        f"{r.last_presented.get('dispatches')}" if r.last_presented else "-"))
    row("Metal 绘制 outcomes（最后）", lambda r: (
        f"{r.draw_outcomes.get('draws')} ran={r.draw_outcomes.get('ran')}" if r.draw_outcomes else "-"))

    def busy(r):
        if not r.busy:
            return "-"
        wall = sum(w for w, _ in r.busy)
        gpu = sum(g for _, g in r.busy)
        return f"{100 * gpu / wall:.0f}%（{len(r.busy)} 段）" if wall else "-"
    row("GPU 忙碌占比", busy)
    row("section 视图（个/总 MB）", lambda r: (f"{r.sections.get('n')} / {r.sections.get('total_mb')}"
                                         if r.sections else "-"))
    row("prewire 成功/失败/超上限/已释放", lambda r: "{wired}/{failed}/{capped}/{freed}，{prewired_mb} MB".format(**r.prewire)
        if any(r.prewire.values()) else "-")
    row("span（个/直接/镜像/MB/capped/refused/stale/stale_uses）", lambda r: "/".join(
        str(r.span.get(k, "-")) for k in ("spans", "direct_spans", "mirror_spans", "span_mb", "capped", "refused", "stale_spans", "stale_uses"))
        if r.span else "-")
    row("截图（帧/final-draw）", lambda r: "{} / {}".format(*map(len, r.captures())))

    out = ["| 项目 | " + " | ".join(cell(r.name) for r in runs) + " |",
           "|---|" + "---|" * len(runs)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    out.append("")
    for r in runs:
        out.append(f"### {r.name}")
        out.append(f"- 目录：`{r.path}`")
        s = r.start_3d()
        if s:
            out.append(f"- 3D 判定：`{s[2]}`")
        for line in r.stuck:
            out.append(f"- 主机阶段卡住：`{line}`")
        for line in r.counts["fuse_lines"]:
            out.append(f"- 熔断：`{line}`")
        for line in r.crash[-5:]:
            out.append(f"- 崩溃/退出线索：`{line}`")
        for line in r.system_gpu[:3]:
            out.append(f"- system-gpu.log：`{line[:200]}`")
        for p in r.hang_samples:
            out.append(f"- hang sample：`{p}`")
        frames, finals = r.captures()
        if frames:
            from_frame = s[1] if s else None
            pick = [p for p in frames if from_frame is not None and int(p.name[5:11]) >= from_frame]
            label = "3D 截图" if pick else "截图（未进 3D，取最后几张）"
            pick = pick or frames
            shown = pick[:png_limit // 2] + pick[-(png_limit - png_limit // 2):] if len(pick) > png_limit else pick
            out.append(f"- {label}（共 {len(pick)} 张，`-rtN.png` 在同目录）：")
            out += [f"  - `{p}`" for p in shown]
        sheet = r.path / "window" / "sheet.png"
        if sheet.exists():
            out.append(f"- 窗口截图拼图：`{sheet}`")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="check-intro.sh run directories")
    ap.add_argument("--3d-frame", dest="three_d_frame", type=int, help="3D starts at the first frame >= N")
    ap.add_argument("--png-limit", type=int, default=6, help="capture paths listed per run")
    ap.add_argument("--json", action="store_true", help="also dump the raw numbers as JSON to stderr")
    args = ap.parse_args()
    runs = [Run(p, args.three_d_frame) for p in args.runs]
    print(report(runs, args.png_limit))
    if args.json:
        json.dump({r.name: dict(counts=r.counts, three_d=r.start_3d(), fps=r.fps_3d(),
                                frames=r.frame_stats(), span=r.span) for r in runs},
                  sys.stderr, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
