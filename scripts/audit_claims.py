#!/usr/bin/env python3
# Claim audit computations (2026-08-15): the exact buffer-conflict
# event count, and the 288-block reference-build epoch and fastest
# event, all from committed per-event data. These back specific paper
# sentences; rerunning must reproduce the printed values exactly.
import csv
import struct


def pct(v, q):
    v = sorted(v)
    return v[min(int(q * len(v)), len(v) - 1)]


total = 0
print("buffer-conflict events (claimed, colliding site, non-poison):")
for part, base, cfgs in [
        ("A10", "results/raw/review2/mid",
         ["stage_drain", "stage_naive", "issue_drain", "issue_naive"]),
        ("A100", "results/raw/a100/mid",
         ["stage_drain", "stage_naive", "issue_naive"])]:
    for cfg in cfgs:
        rows = [r for r in csv.DictReader(open(f"{base}_{cfg}.csv"))
                if int(r["claimer"]) >= 0]
        coll = [r for r in rows
                if (int(r["site"]) == 1 if cfg.startswith("stage")
                    else int(r["site"]) in (0, 1, 2))]
        bad = sum(1 for r in coll if r["corrupt"] == "1")
        print(f"  {part} {cfg}: colliding={len(coll)} corrupt={bad}")
        total += len(coll)
print(f"TOTAL: {total}  (paper: 28,859; zero corrupt)")

with open("results/raw/review2/obs_lat_b288.bin", "rb") as f:
    magic, ev, blk = struct.unpack("<3q", f.read(24))
    setgt = struct.unpack(f"<{ev}q", f.read(8 * ev))
    obs = struct.unpack(f"<{ev * blk}q", f.read(8 * ev * blk))
first, lat = [], []
for e in range(100, ev):
    if setgt[e] == 0:
        continue
    raw = obs[e * blk:(e + 1) * blk]
    if 0 in raw:
        continue
    d = [(raw[b] - setgt[e]) / 1e3 for b in range(1, blk)]
    first.append(min(d))
    lat.append(max(d))
print(f"288 reference build: first-obs p50/p99 {pct(first,.5):.2f}/"
      f"{pct(first,.99):.2f}; lat p50/p99 {pct(lat,.5):.2f}/"
      f"{pct(lat,.99):.2f}; fastest {min(lat):.2f}")
