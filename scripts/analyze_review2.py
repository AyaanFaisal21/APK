#!/usr/bin/env python3
# Review-2 batch analysis, run on the box next to the data.
# Deterministic: fixed inputs, nearest-rank percentiles, fixed formatting.
import csv
import statistics
import struct
import bisect

Q = 1.024  # globaltimer quantum, us


def pct(v, q):
    v = sorted(v)
    return v[min(int(q * len(v)), len(v) - 1)]


# --- atomic surface vs committed volatile-build surface ---------------
old = {(int(r["blocks"]), int(r["poll_every"])): r
       for r in csv.DictReader(open("results/summary/phase4_surface.csv"))}
print(f"{'b':>4} {'p':>2} | lat50 old->new | lat99 old->new | ovh old->new")
worst_dq = 0.0
for b in [16, 36, 72, 144, 216, 288]:
    for p in [1, 2, 4, 8]:
        rows = [r for r in csv.DictReader(open(f"results/raw/review2/lat_b{b}_p{p}.csv"))
                if r["complete"] == "1" and int(r["event"]) >= 100]
        l50 = pct([float(r["lat_excl_setter_us"]) for r in rows], .5)
        l99 = pct([float(r["lat_excl_setter_us"]) for r in rows], .99)
        ov = list(csv.DictReader(open(f"results/raw/review2/ovh_b{b}_p{p}.csv")))
        on = statistics.median([float(r["wall_ms"]) for r in ov if r["poll"] == "1"])
        off = statistics.median([float(r["wall_ms"]) for r in ov if r["poll"] == "0"])
        o = (on / off - 1) * 100
        orow = old[(b, p)]
        worst_dq = max(worst_dq, abs(l50 - float(orow["lat_excl0_p50_us"])) / Q)
        if p == 1:
            print(f"{b:>4} {p:>2} | {float(orow['lat_excl0_p50_us']):6.2f}->{l50:6.2f}"
                  f" | {float(orow['lat_excl0_p99_us']):6.2f}->{l99:6.2f}"
                  f" | {float(orow['overhead_pct']):5.2f}->{o:5.2f}")
print(f"worst p50 delta, all 24 cells: {worst_dq:.1f} quanta")

# --- iid order-statistics test, 288-block obs matrix ------------------
with open("results/raw/review2/obs_lat_b288.bin", "rb") as f:
    magic, ev, blk = struct.unpack("<3q", f.read(24))
    setgt = struct.unpack(f"<{ev}q", f.read(8 * ev))
    obs = struct.unpack(f"<{ev * blk}q", f.read(8 * ev * blk))
delays, dmax, pairs_same, pairs_adj = [], [], [], []
for e in range(100, ev):
    if setgt[e] == 0:
        continue
    raw = obs[e * blk:(e + 1) * blk]
    if 0 in raw:
        continue
    row = [(raw[b] - setgt[e]) / 1e3 for b in range(1, blk)]  # excl setter
    delays.extend(row)
    dmax.append(max(row))
    pairs_same.append((row[10], row[10 + 72]))  # same SM if breadth-first
    pairs_adj.append((row[10], row[11]))        # neighbor SM
delays.sort()
N = blk - 1


def Fpow(x):
    return (bisect.bisect_right(delays, x) / len(delays)) ** N


med_pred = next((i * 0.128 for i in range(1, 2000) if Fpow(i * 0.128) >= 0.5), -1)
print(f"iid-model Dmax p50 {med_pred:.1f} us | empirical Dmax p50 {pct(dmax,.5):.1f}"
      f" | p99 {pct(dmax,.99):.1f} | n={len(dmax)}")


def corr(ps):
    xs = [a for a, b in ps]
    ys = [b for a, b in ps]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((a - mx) * (b - my) for a, b in ps) / len(ps)
    sx = (sum((a - mx) ** 2 for a in xs) / len(xs)) ** .5
    sy = (sum((b - my) ** 2 for b in ys) / len(ys)) ** .5
    return cov / (sx * sy) if sx * sy > 0 else 0.0


print(f"corr(b10,b82 same-SM-if-breadth-first) {corr(pairs_same):.3f}"
      f" | corr(b10,b11 diff-SM) {corr(pairs_adj):.3f}")

# --- summary CSV for the paper pipeline -------------------------------
with open("results/summary/review2_surface.csv", "w") as f:
    f.write("blocks,poll_every,lat_excl0_p50_us,lat_excl0_p99_us,overhead_pct\n")
    for b in [16, 36, 72, 144, 216, 288]:
        for p in [1, 2, 4, 8]:
            rows = [r for r in csv.DictReader(open(f"results/raw/review2/lat_b{b}_p{p}.csv"))
                    if r["complete"] == "1" and int(r["event"]) >= 100]
            l50 = pct([float(r["lat_excl_setter_us"]) for r in rows], .5)
            l99 = pct([float(r["lat_excl_setter_us"]) for r in rows], .99)
            ov = list(csv.DictReader(open(f"results/raw/review2/ovh_b{b}_p{p}.csv")))
            on = statistics.median([float(r["wall_ms"]) for r in ov if r["poll"] == "1"])
            off = statistics.median([float(r["wall_ms"]) for r in ov if r["poll"] == "0"])
            f.write(f"{b},{p},{l50:.3f},{l99:.3f},{(on/off-1)*100:.3f}\n")
print("wrote results/summary/review2_surface.csv")
