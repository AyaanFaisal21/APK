#!/usr/bin/env bash
# Phase 2: urgent-task wait baseline across the three occupancy shapes.
# sat is the true baseline (every residency slot occupied); sat-1 and 1persm
# show what "busy GPU" means when residency is not actually saturated.
set -euo pipefail

TRIALS=${1:-500}
mkdir -p results/raw

for MODE in sat sat-1 1persm; do
  echo ""
  echo "=============== mode=${MODE} ==============="
  ./bin/urgent_baseline --mode "$MODE" --trials "$TRIALS" \
    --csv "results/raw/phase2_${MODE}.csv"
done
