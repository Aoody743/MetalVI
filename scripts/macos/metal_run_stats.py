#!/usr/bin/env python3
"""Join Metal operation, command range, and GPU completion logs."""

import argparse
from collections import Counter
import heapq
import json
import math
import re


ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELDS = re.compile(r'\b(\w+)=(Some\(\[[^]]*\]\)|Some\(\d+\)|\[[^]]*\]|"[^"]*"|[^ ]+)')


def fields(message):
    result = {}
    for key, value in FIELDS.findall(message.rstrip()):
        if value.startswith("Some("):
            value = value[5:-1]
        if value in ("None", "true", "false") or value.isdecimal() or value.startswith(("[", '"')):
            try:
                result[key] = None if value == "None" else json.loads(value)
                continue
            except ValueError:
                pass
        result[key] = value
    return result


def summarize(lines, from_frame=0):
    recorded, pending = {}, {}
    batch = None
    timings, top = [], []
    unmatched = Counter()
    frame = cache = None
    auxiliary = fuses = hangs = 0
    for number, raw in enumerate(lines, 1):
        line = ANSI.sub("", raw)
        if "Metal: recorded operation " in line:
            op = fields(line.partition("Metal: recorded operation ")[2])
            if op["batch"] != batch:
                recorded.clear()
                batch = op["batch"]
            if "indirect" in op["kind"]:
                op["indirect_arguments"] = []
            recorded[op["op_index"]] = op
        elif "Metal: indirect trace record " in line:
            args = fields(line.partition("Metal: indirect trace record ")[2])
            op = recorded.get(args.pop("op_index")) if args.pop("batch") == batch else None
            if op is not None:
                op["indirect_arguments"].append(args)
        elif "Metal: command buffer operations " in line:
            cb = fields(line.partition("Metal: command buffer operations ")[2])
            pending[cb["command"]] = [
                recorded.get(i) if cb["batch"] == batch else None
                for i in range(cb["first"], cb["end"])
            ]
        elif "Metal: command buffer completed " in line:
            cb = fields(line.partition("Metal: command buffer completed ")[2])
            ops = pending.pop(cb["command"], None)
            if ops is None:
                if cb["ops"]:
                    unmatched["missing range"] += 1
                else:
                    auxiliary += 1
                continue
            if len(ops) != 1 or not ops[0] or cb["ops"] != 1:
                unmatched["not one matched operation"] += 1
                continue
            op = ops[0]
            if op["frame"] < from_frame:
                continue
            ns = cb["gpu_ns"]
            timings.append(ns)
            row = (ns, number, {**op, "gpu_ns": ns, "estimated_ns": cb["estimated_ns"],
                               "log_line": number})
            if len(top) < 20:
                heapq.heappush(top, row)
            else:
                heapq.heappushpop(top, row)
        elif "Metal safety stop: submission failed" in line:
            fuses += 1
        elif "has not completed (GPU hang?)" in line:
            hangs += 1
        elif "frame presented (since the last line:" in line:
            frame = fields(line)
        elif "shaders from the cache and translated so far " in line:
            cache = fields(line)
    timings.sort()
    percentile = lambda p: timings[math.ceil(len(timings) * p) - 1] if timings else None
    return {
        "from_frame": from_frame, "timed_single_operation_buffers": len(timings),
        "p50_ns": percentile(0.5), "p99_ns": percentile(0.99), "max_ns": percentile(1),
        "auxiliary_buffers": auxiliary, "unmatched": dict(unmatched),
        "uncompleted_ranges": len(pending), "fuse_log_lines": fuses, "hang_log_lines": hangs,
        "last_present_report": frame, "last_cache_report": cache,
        "top_20": [{k: v for k, v in row[2].items() if k != "indirect_arguments" or v}
                   for row in sorted(top, reverse=True)],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--from-frame", type=int, default=0)
    args = parser.parse_args()
    with open(args.log, encoding="utf-8", errors="replace") as log:
        print(json.dumps(summarize(log, args.from_frame), indent=2))
