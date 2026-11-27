#!/usr/bin/env python3
"""Summarize shader diagnostic counters (SLOT_DIAG) from a game.log.

Usage: metal_diag_stats.py <game.log>

Prints half-wave sync timeouts by pipeline and site, and loop-budget stops by
pipeline, as logged by victory-gpu's backend/metal_diag.rs after each command
buffer ("Metal: half-wave sync timed out", "Metal: a loop budget stopped").
"""
import collections
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r'(\w+)=("([^"]*)"|\S+)')


def fields(line):
    out = {}
    for m in FIELD.finditer(line):
        out[m.group(1)] = m.group(3) if m.group(3) is not None else m.group(2)
    return out


def pipelines(ops):
    return sorted(set(re.findall(r"[0-9a-f]{16}", ops or ""))) or [ops or "?"]


def main(path):
    timeouts = collections.Counter()
    sites = collections.defaultdict(collections.Counter)
    buffers = collections.Counter()
    budget = collections.Counter()
    budget_lines = 0
    with open(path, "rb") as f:
        for raw in f:
            line = ANSI.sub("", raw.decode("utf-8", "replace"))
            if "half-wave sync timed out" in line:
                f_ = fields(line)
                n = int(f_.get("timeouts", 0))
                for p in pipelines(f_.get("ops")):
                    timeouts[p] += n
                    buffers[p] += 1
                    for s in filter(None, f_.get("sites", "").split(",")):
                        site, count = s.split(":")
                        sites[p][int(site)] += int(count)
            elif "a loop budget stopped" in line:
                f_ = fields(line)
                budget_lines += 1
                for p in pipelines(f_.get("ops")):
                    budget[p] += int(f_.get("waves", 0))
    print(f"half-wave sync timeouts: {sum(timeouts.values())} host waves")
    for p, n in timeouts.most_common():
        top = ", ".join(f"{s}:{c}" for s, c in sorted(sites[p].items()))
        print(f"  {p}: {n} in {buffers[p]} command buffers; sites {top}")
    print(f"loop-budget stops: {sum(budget.values())} waves ({budget_lines} lines logged; the log is rate-limited after 64)")
    for p, n in budget.most_common(20):
        print(f"  {p}: {n}")


if __name__ == "__main__":
    main(sys.argv[1])
