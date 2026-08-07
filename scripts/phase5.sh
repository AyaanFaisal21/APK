#!/usr/bin/env bash
# Phase 5: mid-pipeline yield safety. Verify gate first (drain must pass,
# poison must trip the detector), then 10k-event safety runs per
# discipline, the boundary-poll latency comparator, and the overhead trio.
set -euo pipefail

EVENTS=${1:-10000}
mkdir -p results/raw

echo "=============== verify: drain (must PASS) ==============="
./bin/midyield --mode verify --discipline drain --tasks 20000
echo ""
echo "=============== verify: poison (detector control, must PASS) ==============="
./bin/midyield --mode verify --discipline poison --tasks 20000

for D in drain naive poison; do
  echo ""
  echo "=============== events: discipline=$D, stage poll, sat ==============="
  ./bin/midyield --mode events --discipline "$D" --events "$EVENTS" \
    --csv "results/raw/phase5_${D}.csv"
done

echo ""
echo "=============== events: drain, boundary poll (latency comparator) ==============="
./bin/midyield --mode events --discipline drain --poll boundary \
  --events "$EVENTS" --csv "results/raw/phase5_drain_boundary.csv"

for P in stage boundary; do
  echo ""
  echo "=============== overhead: poll=$P vs off ==============="
  ./bin/midyield --mode overhead --poll "$P" --tasks 200000 --reps 20
done
