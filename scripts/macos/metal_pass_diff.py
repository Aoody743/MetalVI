#!/usr/bin/env python3
"""T2 per-pass screenshot diff of two check-intro.sh runs.

    metal_pass_diff.py RUN_A RUN_B [FRAME] [--tolerance N] [--diff-dir DIR] [--no-pil]

Compares RUN_A/frames/frameNNNNNN.png and every frameNNNNNN-rtK.png with
RUN_B's. Without FRAME: one summary line for every frame captured in both.

What the captures are (crates/victory-gpu/src/queue.rs, `present`): every
VICTORY_METAL_CAPTURE_EVERY frames the presented texture (frameNNNNNN.png)
and the six largest colour targets Metal drew into since the last capture
(-rt0 = largest; ties in first-draw order). check-intro.sh shrinks every
capture to fit 640x360, so compare runs taken with the same check-intro.

Pairing: rtK of A with rtK of B when both have the same size and format
(from the "Metal: capturing a texture" lines in game.log); otherwise the
first unpaired B target with the same size and format, else by name.
Order of passes: logs since v3-tools have "a render target's place in the
draw order" (`order` = first draw of the target since the last capture);
"first differing pass" is the lowest such order whose pixels differ.
Older logs lack it: the report then orders by rt index (largest first)
and says so.

A pixel differs when any channel differs by more than --tolerance
(default 0: exact). --diff-dir writes a heat map per differing pair.
Images: PIL (+ numpy when present); --no-pil or a missing PIL converts
through macOS `sips` to BMP and reads that.
"""

import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
CAPTURE = re.compile(r'Metal: capturing a texture name="(frame\d{6}(?:-rt\d+)?)" resource=(\d+) w=(\d+) h=(\d+) format=(\d+)')
ORDER = re.compile(r"Metal capture: a render target's place in the draw order frame=(\d+) rt=(\d+) resource=(\d+) order=(\d+) targets=(\d+)")

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


# ---- image loading -------------------------------------------------------

def read_bmp(path):
    data = Path(path).read_bytes()
    if data[:2] != b"BM":
        raise ValueError(f"{path}: not a BMP")
    offset = struct.unpack_from("<I", data, 10)[0]
    w, h, _, bpp = struct.unpack_from("<iiHH", data, 18)
    if bpp not in (24, 32):
        raise ValueError(f"{path}: {bpp}-bit BMP")
    flip, h = h > 0, abs(h)
    step = bpp // 8
    stride = (w * step + 3) & ~3
    rows = []
    for y in range(h):
        row = data[offset + y * stride: offset + y * stride + w * step]
        rows.append([(row[x * step + 2], row[x * step + 1], row[x * step]) for x in range(w)])
    if flip:
        rows.reverse()
    return w, h, [p for r in rows for p in r]


class Picture:
    """RGB pixels as numpy (h, w, 3) or a flat list of tuples."""

    def __init__(self, path, use_pil=True):
        self.path = path
        if use_pil and Image is not None:
            with Image.open(path) as im:
                im = im.convert("RGB")
                self.size = im.size
                self.pixels = np.asarray(im, dtype=np.int16) if np is not None else list(im.getdata())
        else:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "x.bmp")
                subprocess.run(["sips", "-s", "format", "bmp", str(path), "--out", out],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                w, h, px = read_bmp(out)
            self.size = (w, h)
            self.pixels = px


def compare(a, b, tolerance):
    """(differing pixels, total, max channel difference, mean abs difference, mask or None)."""
    if a.size != b.size:
        return None
    w, h = a.size
    if np is not None and not isinstance(a.pixels, list) and not isinstance(b.pixels, list):
        d = np.abs(a.pixels - b.pixels)
        m = d.max(axis=2)
        mask = m > tolerance
        return int(mask.sum()), w * h, int(m.max()), float(d.mean()), m
    diff = 0
    mx = 0
    total = 0
    m = []
    for pa, pb in zip(a.pixels, b.pixels):
        c = max(abs(x - y) for x, y in zip(pa, pb))
        total += sum(abs(x - y) for x, y in zip(pa, pb))
        m.append(c)
        mx = max(mx, c)
        if c > tolerance:
            diff += 1
    return diff, w * h, mx, total / (3 * w * h), m


def heat_map(mask, size, path):
    if Image is None:
        return False
    w, h = size
    if np is not None and not isinstance(mask, list):
        v = np.clip(mask.astype(np.int32) * 4, 0, 255).astype(np.uint8)
        rgb = np.stack([v, np.zeros_like(v), 255 - v], axis=2)
        rgb[mask == 0] = 0
        Image.fromarray(rgb).save(path)
    else:
        im = Image.new("RGB", (w, h))
        im.putdata([(0, 0, 0) if c == 0 else (min(255, c * 4), 0, 255 - min(255, c * 4)) for c in mask])
        im.save(path)
    return True


# ---- run side ------------------------------------------------------------

class Run:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.captures = {}   # name -> (resource, w, h, format)
        self.order = {}      # (frame, rt) -> (order, targets, resource)
        log = self.path / "game.log"
        if log.exists():
            with open(log, errors="replace") as f:
                for raw in f:
                    if "Metal capture" not in raw and "capturing a texture" not in raw:
                        continue
                    line = ANSI.sub("", raw)
                    m = CAPTURE.search(line)
                    if m:
                        self.captures[m.group(1)] = tuple(int(x) for x in m.group(2, 3, 4, 5))
                        continue
                    m = ORDER.search(line)
                    if m:
                        frame, rt, res, order, targets = (int(x) for x in m.groups())
                        self.order[(frame, rt)] = (order, targets, res)

    def frames(self):
        d = self.path / "frames"
        return sorted(int(p.name[5:11]) for p in d.glob("frame*.png") if re.fullmatch(r"frame\d{6}\.png", p.name))

    def targets(self, frame):
        d = self.path / "frames"
        out = {}
        for p in d.glob(f"frame{frame:06d}-rt*.png"):
            m = re.fullmatch(r"frame\d{6}-rt(\d+)\.png", p.name)
            if m:
                out[int(m.group(1))] = p
        return dict(sorted(out.items()))

    def info(self, frame, rt=None):
        name = f"frame{frame:06d}" + ("" if rt is None else f"-rt{rt}")
        return self.captures.get(name)


def describe(info):
    if not info:
        return "?"
    _, w, h, fmt = info
    return f"{w}x{h} f{fmt}"


def pair_targets(a, b, frame):
    ta, tb = a.targets(frame), b.targets(frame)
    pairs, left = [], dict(tb)
    for k, pa in ta.items():
        ia = a.info(frame, k)
        key = ia[1:] if ia else None
        ib = b.info(frame, k)
        if k in left and (key is None or ib is None or ib[1:] == key):
            pairs.append((k, k, pa, left.pop(k)))
            continue
        match = next((kb for kb in left if key and b.info(frame, kb) and b.info(frame, kb)[1:] == key), None)
        if match is not None:
            pairs.append((k, match, pa, left.pop(match)))
        else:
            pairs.append((k, None, pa, None))
    pairs += [(None, kb, None, pb) for kb, pb in left.items()]
    return pairs


def diff_frame(a, b, frame, args, verbose=True):
    rows = []
    fa, fb = a.path / "frames" / f"frame{frame:06d}.png", b.path / "frames" / f"frame{frame:06d}.png"
    entries = [("present", None, None, fa if fa.exists() else None, fb if fb.exists() else None)]
    entries += [(f"rt{ka}" if ka is not None else f"(B rt{kb})", ka, kb, pa, pb) for ka, kb, pa, pb in pair_targets(a, b, frame)]
    for label, ka, kb, pa, pb in entries:
        ia = a.info(frame, ka) if pa else None
        ib = b.info(frame, kb) if pb else None
        oa = a.order.get((frame, ka)) if ka is not None else None
        ob = b.order.get((frame, kb)) if kb is not None else None
        row = dict(label=label, ka=ka, kb=kb, ia=ia, ib=ib, oa=oa, ob=ob, pa=pa, pb=pb, result=None, note="")
        if pa is None or pb is None:
            row["note"] = "只在 " + ("B" if pa is None else "A") + " 里有"
        else:
            if ia and ib and ia[1:] != ib[1:]:
                row["note"] = f"尺寸/格式不同 {describe(ia)} vs {describe(ib)}"
            pic_a, pic_b = Picture(pa, not args.no_pil), Picture(pb, not args.no_pil)
            r = compare(pic_a, pic_b, args.tolerance)
            if r is None:
                row["note"] = (row["note"] + "; " if row["note"] else "") + f"PNG 尺寸不同 {pic_a.size} vs {pic_b.size}"
            else:
                row["result"] = r[:4]
                if r[0] and args.diff_dir:
                    os.makedirs(args.diff_dir, exist_ok=True)
                    out = Path(args.diff_dir) / f"{a.path.name}--{b.path.name}-frame{frame:06d}-{label.strip('()').replace(' ', '')}.png"
                    if heat_map(r[4], pic_a.size, out):
                        row["heat"] = str(out)
        rows.append(row)
    return rows


def fmt_rows(a, b, frame, rows):
    have_order = any(r["oa"] is not None for r in rows)
    out = [f"## 帧 {frame}：`{a.path.name}`（A） vs `{b.path.name}`（B）", "",
           "| 图 | A 资源 / 尺寸 | B 资源 / 尺寸 | 绘制顺序 A/B | 不同像素 | 占比 | 最大差 | 平均差 | 备注 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ia, ib = r["ia"], r["ib"]
        oa = f"{r['oa'][0]}/{r['oa'][1]}" if r["oa"] else "-"
        ob = f"{r['ob'][0]}/{r['ob'][1]}" if r["ob"] else "-"
        if r["result"]:
            n, total, mx, mean = r["result"]
            cells = [str(n), f"{100 * n / total:.2f}%", str(mx), f"{mean:.2f}"]
        else:
            cells = ["-"] * 4
        note = r["note"] + (f" 热图 `{r['heat']}`" if r.get("heat") else "")
        out.append("| {} | {} | {} | {} | {} | {} |".format(
            r["label"], f"{ia[0]} / {describe(ia)}" if ia else "-", f"{ib[0]} / {describe(ib)}" if ib else "-",
            f"{oa} vs {ob}", " | ".join(cells), note))
    differing = [r for r in rows if r["label"] != "present" and (r["result"] is None or r["result"][0])]
    if have_order:
        key = lambda r: (r["oa"] or r["ob"] or (1 << 30, 0))[0]
        how = "按绘制顺序（首次绘制）"
    else:
        key = lambda r: r["ka"] if r["ka"] is not None else 99
        how = "日志里没有绘制顺序（旧日志），按 rt 序号（面积从大到小）排，不代表 pass 先后"
    out.append("")
    if differing:
        first = min(differing, key=key)
        out.append(f"**第一个不同的渲染目标**：{first['label']}（{how}）。")
    else:
        same = all(r["result"] and r["result"][0] == 0 for r in rows)
        out.append("**所有渲染目标一致**。" if same else "渲染目标没有可比较的不同（见备注）。")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("frame", nargs="?", type=int, help="frame number; omitted: summary of every common frame")
    ap.add_argument("--tolerance", type=int, default=0, help="per-channel difference still counted as equal")
    ap.add_argument("--diff-dir", help="write heat maps of differing pairs here")
    ap.add_argument("--no-pil", action="store_true", help="read images through sips (fallback path)")
    args = ap.parse_args()
    a, b = Run(args.run_a), Run(args.run_b)
    if args.frame is not None:
        if not (a.path / "frames" / f"frame{args.frame:06d}.png").exists() and not a.targets(args.frame):
            sys.exit(f"{a.path.name}: no capture of frame {args.frame} (have {a.frames()[:20]})")
        print(fmt_rows(a, b, args.frame, diff_frame(a, b, args.frame, args)))
        return
    common = sorted(set(a.frames()) & set(b.frames()))
    if not common:
        sys.exit(f"no common captured frame (A: {a.frames()[:10]}..., B: {b.frames()[:10]}...)")
    print(f"| 帧 | present 不同像素 | 不同的 rt（不同像素占比） |\n|---|---|---|")
    for frame in common:
        rows = diff_frame(a, b, frame, args)
        pres = next(r for r in rows if r["label"] == "present")
        p = f"{100 * pres['result'][0] / pres['result'][1]:.2f}%" if pres["result"] else pres["note"] or "-"
        bad = [f"{r['label']} {100 * r['result'][0] / r['result'][1]:.1f}%" if r["result"] else f"{r['label']} {r['note']}"
               for r in rows if r["label"] != "present" and (r["result"] is None or r["result"][0])]
        print(f"| {frame} | {p} | {', '.join(bad) or '一致'} |")


if __name__ == "__main__":
    main()
