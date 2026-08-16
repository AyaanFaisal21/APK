#!/usr/bin/env python3
# Held-out max-of-N analysis (review 3, point 5) and every-event floor
# check (point 2), computed from the released observation matrices.
# Deterministic: fixed seed, fixed split (first half fits, second half
# tests), nearest-rank percentiles.
import struct
import bisect
import random
import os


def load(path, warm=100, excl_setter=True):
    with open(path, "rb") as f:
        magic, ev, blk = struct.unpack("<3q", f.read(24))
        setgt = struct.unpack(f"<{ev}q", f.read(8 * ev))
        obs = struct.unpack(f"<{ev * blk}q", f.read(8 * ev * blk))
    lo = 1 if excl_setter else 0
    rows = []
    for e in range(warm, ev):
        if setgt[e] == 0:
            continue
        raw = obs[e * blk:(e + 1) * blk]
        if 0 in raw:
            continue
        rows.append([(raw[b] - setgt[e]) / 1e3 for b in range(lo, blk)])
    return rows


def pct(v, q):
    v = sorted(v)
    return v[min(int(q * len(v)), len(v) - 1)]


PARTS = [("A10-288", "results/raw/review2/obs_lat_b288.bin"),
         ("A100-432", "results/raw/a100/obs_lat_b432.bin")]
FLOORS = [("A10-288", "results/raw/review2/obs_visfloor_b288.bin"),
          ("A100-432", "results/raw/a100/visf_b432.bin")]

os.makedirs("results/summary", exist_ok=True)
out = open("results/summary/heldout_maxofn.csv", "w")
out.write("part,quantile,predicted_us,measured_us\n")

print("=== floor visibility: every-event check ===")
for name, p in FLOORS:
    rows = load(p)
    allmax = max(max(r) for r in rows)
    over = sum(1 for r in rows if max(r) > 1.03)
    print(f"{name}: n={len(rows)} absolute max {allmax:.2f} us, "
          f"events over one quantum: {over}")

print("=== held-out independence model (fit half A, test half B) ===")
for name, p in PARTS:
    rows = load(p)
    N = len(rows[0])
    half = len(rows) // 2
    marg = sorted(d for r in rows[:half] for d in r)
    dmaxB = [max(r) for r in rows[half:]]

    def Fmax(x):
        return (bisect.bisect_right(marg, x) / len(marg)) ** N

    def q_pred(q):
        lo, hi = 0.0, 4000.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if Fmax(mid) < q:
                lo = mid
            else:
                hi = mid
        return hi

    print(f"{name} (N={N}, fit n={half}, test n={len(dmaxB)}):")
    for q in (.5, .9, .95, .99):
        pr, me = q_pred(q), pct(dmaxB, q)
        print(f"  p{int(q*100):02d}: predicted {pr:7.2f}  measured {me:7.2f}")
        out.write(f"{name},{q},{pr:.2f},{me:.2f}\n")

    perblk = [pct([r[b] for r in rows], .5) for b in range(N)]
    print(f"  per-block median delays: min {min(perblk):.2f} "
          f"p50 {pct(perblk, .5):.2f} max {max(perblk):.2f}")
    random.seed(21)
    cors = []
    for _ in range(300):
        i, j = random.sample(range(N), 2)
        xi = [r[i] for r in rows]
        xj = [r[j] for r in rows]
        mi, mj = sum(xi) / len(xi), sum(xj) / len(xj)
        cov = sum((a - mi) * (b - mj) for a, b in zip(xi, xj)) / len(xi)
        si = (sum((a - mi) ** 2 for a in xi) / len(xi)) ** .5
        sj = (sum((b - mj) ** 2 for b in xj) / len(xj)) ** .5
        if si * sj > 0:
            cors.append(cov / (si * sj))
    print(f"  pairwise corr, 300 pairs: min {min(cors):.3f} "
          f"p50 {pct(cors, .5):.3f} max {max(cors):.3f}")
out.close()
print("wrote results/summary/heldout_maxofn.csv")
