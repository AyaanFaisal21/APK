#!/usr/bin/env bash
# Phase 4: the surface. Resident block count x polling period, two outputs
# per cell (preemption latency distribution, steady-state overhead).
# Latency: dev-set (calibration-free), 10k events + 100 warmup per cell.
# Overhead: ABAB drains, tasks scaled with blocks so per-rep wall stays
# ~constant (~700 tasks/block).
set -euo pipefail

EVENTS=${1:-10000}
BLOCKS_LIST="16 36 72 144 216 288"
POLL_LIST="1 2 4 8"
mkdir -p results/raw

for B in $BLOCKS_LIST; do
  for P in $POLL_LIST; do
    echo ""
    echo "=============== blocks=$B poll-every=$P: latency ==============="
    ./bin/yield --mode latency --set dev --events "$EVENTS" \
      --blocks "$B" --poll-every "$P" \
      --csv "results/raw/phase4_lat_b${B}_p${P}.csv"
    echo "--------------- blocks=$B poll-every=$P: overhead ---------------"
    ./bin/yield --mode overhead --tasks $((B * 700)) --reps 20 \
      --blocks "$B" --poll-every "$P" \
      --csv "results/raw/phase4_ovh_b${B}_p${P}.csv"
  done
done
