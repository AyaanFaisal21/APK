#!/usr/bin/env python3
# The paper's main figure, generated from committed summary CSVs only.
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.0))

# (a) latency vs blocks per poll period — A10 atomic build
rows = list(csv.DictReader(open("results/summary/review2_surface.csv")))
colors = {1: "#1a6faf", 2: "#4d9e4d", 4: "#d88a2d", 8: "#b04a4a"}
for p in [1, 2, 4, 8]:
    pts = [(int(r["blocks"]), float(r["lat_excl0_p50_us"]))
           for r in rows if int(r["poll_every"]) == p]
    ax1.plot([x for x, _ in pts], [y for _, y in pts], "o-",
             color=colors[p], label=f"poll every {p}")
ax1.set_xlabel("resident blocks (A10, 72 SMs)")
ax1.set_ylabel("handoff latency p50 (µs)")
ax1.set_yscale("log")
ax1.set_xticks([16, 36, 72, 144, 216, 288])
ax1.legend(fontsize=8)
ax1.set_title("(a) Latency: residual work scales,\nnot propagation", fontsize=9)
ax1.grid(alpha=0.25)

# (b) notification tail: floor vs loaded, both parts
vis = list(csv.DictReader(open("results/summary/visibility.csv")))
styles = {("A10", "floor"): ("#1a6faf", "o-"),
          ("A10", "loaded"): ("#1a6faf", "s--"),
          ("A100", "floor"): ("#b04a4a", "o-"),
          ("A100", "loaded"): ("#b04a4a", "s--")}
for (part, var), (c, st) in styles.items():
    pts = sorted((int(r["blocks"]), float(r["dmax_p50_us"]))
                 for r in vis if r["part"] == part and r["variant"] == var)
    if pts:
        ax2.plot([x for x, _ in pts], [y for _, y in pts], st, color=c,
                 label=f"{part} {var}", markersize=4)
ax2.set_xlabel("resident blocks")
ax2.set_ylabel("across-block max notification delay p50 (µs)")
ax2.set_yscale("log")
ax2.axhline(1.024, color="gray", ls=":", lw=1)
ax2.text(150, 1.15, "timer quantum", fontsize=7, color="gray")
ax2.legend(fontsize=8)
ax2.set_title("(b) Notification: free at the floor,\ncongestion-bound under load", fontsize=9)
ax2.grid(alpha=0.25)

# (c) Pareto: throughput loss vs urgent p99 — A10
par = list(csv.DictReader(open("results/summary/pareto.csv")))
res = [r for r in par if r["arm"] == "reserved"]
coop = [r for r in par if r["arm"] == "coop"]
umark = {"1": "o", "4": "^", "16": "s"}
for r in res:
    ax3.scatter(float(r["rate_loss_pct"]), float(r["e2e_p99_us"]),
                marker=umark[r["U"]], c="#7a5aa0", s=40)
for r in coop:
    ax3.scatter(float(r["rate_loss_pct"]), float(r["e2e_p99_us"]),
                marker=umark[r["U"]], c="#12735f", s=60)
for u, m in umark.items():
    ax3.scatter([], [], marker=m, c="gray", label=f"U={u}")
ax3.scatter([], [], c="#7a5aa0", label="reserved (R=1..32)")
ax3.scatter([], [], c="#12735f", label="cooperative")
ax3.set_xlabel("background throughput loss (%)")
ax3.set_ylabel("urgent-job e2e p99 (µs)")
ax3.set_yscale("log")
ax3.legend(fontsize=7, loc="upper right")
ax3.set_title("(c) Reserved capacity vs cooperative\nhandoff (A10)", fontsize=9)
ax3.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("paper/figures/handoff.png", dpi=150)
print("wrote paper/figures/handoff.png")
