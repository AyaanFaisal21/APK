#!/usr/bin/env python3
# Post-registered reanalysis of the existing Phase 4 raw events. No new
# data. Motivated by external review (2026-08-08): first-observer delay
# is a biased construct for propagation scaling; the observation spread
# (last obs - first obs across resident blocks, per event) is on disk
# for every event and characterizes how synchronized the grid's handoff
# is. Full per-block quantiles (D50/D95) need the instrumented rerun;
# this script exhausts what the existing CSVs can say first.
#
# Deterministic: fixed input set, sorted iteration, fixed warmup cut
# (event >= 100), fixed percentile method (nearest-rank), fixed output
# formatting. Rerunning it on the same CSVs produces byte-identical
# output.
import csv
import os
import sys

BLOCKS = [16, 36, 72, 144, 216, 288]
POLL = [1, 2, 4, 8]
RAW = "results/raw"
OUT = "results/summary/spread_scaling.csv"


def pct(v, q):
    v = sorted(v)
    return v[min(int(q * len(v)), len(v) - 1)]


def main():
    rows_out = []
    for b in BLOCKS:
        for p in POLL:
            path = f"{RAW}/phase4_lat_b{b}_p{p}.csv"
            if not os.path.exists(path):
                sys.exit(f"missing {path}: run from repo root with raw data present")
            rs = [r for r in csv.DictReader(open(path))
                  if r["complete"] == "1" and int(r["event"]) >= 100]
            spread = [float(r["spread_us"]) for r in rs]
            first = [(int(r["first_obs_ns"]) - int(r["set_ns"])) / 1e3 for r in rs]
            lat = [float(r["lat_us"]) for r in rs]
            # arithmetic identity check: lat == first_delay + spread exactly
            # (both derive from the same three timestamps); tolerance covers
            # the 0.01 us CSV rounding only. A violation means a parser bug.
            worst_id = max(abs(l - f - s) for l, f, s in zip(lat, first, spread))
            assert worst_id < 0.02, f"identity violated at b={b} p={p}: {worst_id}"
            rows_out.append((b, p, len(rs),
                             pct(spread, 0.50), pct(spread, 0.95), pct(spread, 0.99),
                             pct(first, 0.50), pct(first, 0.99)))

    os.makedirs("results/summary", exist_ok=True)
    with open(OUT, "w") as f:
        f.write("blocks,poll_every,n,spread_p50_us,spread_p95_us,spread_p99_us,"
                "first_p50_us,first_p99_us\n")
        for r in rows_out:
            f.write(",".join(f"{x:.3f}" if isinstance(x, float) else str(x)
                             for x in r) + "\n")

    print(f"{'blocks':>6} {'poll':>4} | {'spr_p50':>7} {'spr_p95':>7} {'spr_p99':>7} | {'first_p50':>9}")
    for r in rows_out:
        print(f"{r[0]:>6} {r[1]:>4} | {r[3]:>7.2f} {r[4]:>7.2f} {r[5]:>7.2f} | {r[6]:>9.2f}")

    # scaling ratios along each axis, poll-every-1 column and 288-block row
    k1 = {r[0]: r for r in rows_out if r[1] == 1}
    b288 = {r[1]: r for r in rows_out if r[0] == 288}
    print("\nspread p50 growth with blocks (poll=1):",
          " ".join(f"{k1[BLOCKS[i+1]][3]/k1[BLOCKS[i]][3]:.2f}x"
                   for i in range(len(BLOCKS) - 1)))
    print("spread p50 growth with poll period (blocks=288):",
          " ".join(f"{b288[POLL[i+1]][3]/b288[POLL[i]][3]:.2f}x"
                   for i in range(len(POLL) - 1)))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
