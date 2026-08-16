#!/usr/bin/env python3
# A3 Pareto analysis, run beside the data. Deterministic.
import csv
import statistics


def pct(v, q):
    v = sorted(v)
    return v[min(int(q * len(v)), len(v) - 1)]


def cell(path):
    rows = list(csv.DictReader(open(path)))
    e2e = [float(r["e2e_us"]) for r in rows]
    drain = [float(r["drain_us"]) for r in rows]
    return pct(e2e, .5), pct(e2e, .99), statistics.median(drain), len(rows)


# reference drain rate: no-reserve cell
_, _, d0, _ = cell("results/raw/pareto/res_R0_U1.csv")
rate0 = 20000 / d0 * 1e6
print(f"reference (R=0) drain rate: {rate0:,.0f} tasks/s "
      f"(median drain {d0:.0f} us)")
print(f"{'arm':>9} {'R':>3} {'U':>3} | {'rate loss %':>11} | "
      f"{'e2e p50':>8} {'e2e p99':>8} | {'n':>6}")
out = [("arm", "R", "U", "rate_loss_pct", "e2e_p50_us", "e2e_p99_us", "n")]
for R in [1, 2, 4, 8, 16, 32]:
    for U in [1, 4, 16]:
        p50, p99, dmed, n = cell(f"results/raw/pareto/res_R{R}_U{U}.csv")
        loss = (1 - (20000 / dmed * 1e6) / rate0) * 100
        print(f"{'reserved':>9} {R:>3} {U:>3} | {loss:>10.2f}  | "
              f"{p50:>8.1f} {p99:>8.1f} | {n:>6}")
        out.append(("reserved", R, U, f"{loss:.3f}", f"{p50:.2f}",
                    f"{p99:.2f}", n))

# cooperative arm: overhead reference for the throughput axis
ov = list(csv.DictReader(open("results/raw/pareto/coop_ovh.csv")))
on = statistics.median([float(r["wall_ms"]) for r in ov if r["poll"] == "1"])
off = statistics.median([float(r["wall_ms"]) for r in ov if r["poll"] == "0"])
coop_loss = (on / off - 1) * 100
for U in [1, 4, 16]:
    rows = [r for r in csv.DictReader(open(f"results/raw/pareto/coop_U{U}.csv"))
            if r["complete"] == "1" and int(r["event"]) >= 100
            and float(r["e2e_us"]) >= 0]
    e2e = [float(r["e2e_us"]) for r in rows]
    # urgent-work capacity fraction at the measured arrival cadence
    # (~275 us mean gap): U tiles x ~25 us across 288 slots
    frac = U * 25.4 / 275.0 / 288 * 100
    loss = coop_loss + frac
    print(f"{'coop':>9} {'-':>3} {U:>3} | {loss:>10.2f}  | "
          f"{pct(e2e,.5):>8.1f} {pct(e2e,.99):>8.1f} | {len(e2e):>6}")
    out.append(("coop", "-", U, f"{loss:.3f}", f"{pct(e2e,.5):.2f}",
                f"{pct(e2e,.99):.2f}", len(e2e)))
print(f"coop poll overhead component: {coop_loss:.2f}% "
      f"(work component computed per U)")

with open("results/summary/pareto.csv", "w") as f:
    for r in out:
        f.write(",".join(str(x) for x in r) + "\n")
print("wrote results/summary/pareto.csv")
