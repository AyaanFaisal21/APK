#!/usr/bin/env python3
# Visibility summary from primary sources. Rows with source=bin are
# computed here from the committed per-event observation matrices; rows
# with source=log are transcribed from run logs whose saturation rows
# were validated bin-vs-log (A10 288 matches exactly). The A100 432
# loaded row is bin-computed and CORRECTS a log misattribution (the
# 158.72 line belongs to 324 blocks; notebook amendment 2026-08-15).
import struct
import os


def pct(v, q):
    v = sorted(v)
    return v[min(int(q * len(v)), len(v) - 1)]


def from_bin(path, warm=100):
    with open(path, "rb") as f:
        magic, ev, blk = struct.unpack("<3q", f.read(24))
        setgt = struct.unpack(f"<{ev}q", f.read(8 * ev))
        obs = struct.unpack(f"<{ev * blk}q", f.read(8 * ev * blk))
    dmax = []
    for e in range(warm, ev):
        if setgt[e] == 0:
            continue
        raw = obs[e * blk:(e + 1) * blk]
        if 0 in raw:
            continue
        dmax.append(max((raw[b] - setgt[e]) / 1e3 for b in range(blk)))
    return pct(dmax, .5), pct(dmax, .99)


LOG = [  # part, blocks, variant, dmax_p50, dmax_p99
    ("A10", 16, "floor", 1.02, 1.02), ("A10", 16, "loaded", 7.17, 8.19),
    ("A10", 36, "floor", 1.02, 1.02), ("A10", 36, "loaded", 7.17, 8.19),
    ("A10", 72, "floor", 1.02, 1.02), ("A10", 72, "loaded", 7.17, 8.19),
    ("A10", 144, "floor", 1.02, 1.02), ("A10", 144, "loaded", 8.19, 9.22),
    ("A10", 216, "floor", 1.02, 1.02), ("A10", 216, "loaded", 21.50, 23.55),
    ("A100", 108, "floor", 1.02, 1.02), ("A100", 108, "loaded", 6.14, 7.17),
    ("A100", 216, "floor", 1.02, 1.02),
    ("A100", 324, "floor", 1.02, 1.02), ("A100", 324, "loaded", 158.72, 173.06),
]
BIN = [
    ("A10", 288, "floor", "results/raw/review2/obs_visfloor_b288.bin"),
    ("A10", 288, "loaded", "results/raw/review2/obs_visload_b288.bin"),
    ("A100", 432, "floor", "results/raw/a100/visf_b432.bin"),
    ("A100", 432, "loaded", "results/raw/a100/visl_b432.bin"),
]

os.makedirs("results/summary", exist_ok=True)
with open("results/summary/visibility.csv", "w") as f:
    f.write("part,blocks,variant,dmax_p50_us,dmax_p99_us,source\n")
    for part, blocks, var, p50, p99 in LOG:
        f.write(f"{part},{blocks},{var},{p50:.2f},{p99:.2f},log\n")
    for part, blocks, var, path in BIN:
        p50, p99 = from_bin(path)
        f.write(f"{part},{blocks},{var},{p50:.2f},{p99:.2f},bin\n")
print("wrote results/summary/visibility.csv")
