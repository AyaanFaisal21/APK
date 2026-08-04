#!/usr/bin/env bash
# Phase 3: the two numbers. Correctness gate first, then steady-state
# polling overhead, then preemption latency >= 10k events in both set modes
# (dev = calibration-free primary; host = CLAIM.md's literal definition).
set -euo pipefail

EVENTS=${1:-10000}
mkdir -p results/raw

echo "=============== verify (yield path, float64) ==============="
./bin/yield --mode verify --tasks 20000

echo ""
echo "=============== overhead: poll on vs off ==============="
./bin/yield --mode overhead --tasks 200000 --reps 20 \
  --csv results/raw/phase3_overhead.csv

echo ""
echo "=============== latency, device-set (primary) ==============="
./bin/yield --mode latency --set dev --events "$EVENTS" \
  --csv results/raw/phase3_lat_dev.csv

echo ""
echo "=============== latency, host-set (CLAIM.md definition) ==============="
./bin/yield --mode latency --set host --events "$EVENTS" \
  --csv results/raw/phase3_lat_host.csv
