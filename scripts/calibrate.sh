#!/usr/bin/env bash
# Phase 1 calibration: sweep K to map tile duration onto the 2-25 us band.
# K is the only knob; everything else stays at defaults (72 blocks, 48x48
# tiles). Verification stays ON for every point — a fast wrong tile is worse
# than a slow run. Raw CSVs land in results/raw/ (gitignored).
#
# Usage: ./scripts/calibrate.sh [SM_MHZ]   (must match the locked clock)
set -euo pipefail

SM_MHZ=${1:-1695}
mkdir -p results/raw

for K in 32 64 128 256 512; do
  echo ""
  echo "=============== K=${K} ==============="
  ./bin/persistent --k "$K" --sm-clock-mhz "$SM_MHZ" \
    --csv "results/raw/calib_k${K}.csv"
done

echo ""
echo "Done. p50s above map K -> tile duration; pick the Ks bracketing 2-25 us."
