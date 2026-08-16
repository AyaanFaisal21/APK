#!/usr/bin/env bash
# A3: reserved-capacity versus cooperative handoff, urgent-size sweep
# included. Reserved arm: R slots held out of the grid, urgent job = U
# blocks on a max-priority stream. Cooperative arm: urgent job = U tiles
# popped by yielding workers. x-axis: bg drain rate; y-axis: urgent e2e.
set -euo pipefail
TRIALS=${1:-10000}
EVENTS=${2:-10000}
mkdir -p results/raw/pareto
for R in 1 2 4 8 16 32; do
  for U in 1 4 16; do
    ./bin/urgent_baseline --mode sat --reserve "$R" --urgent-blocks "$U" \
      --trials "$TRIALS" --csv "results/raw/pareto/res_R${R}_U${U}.csv"
  done
done
# no-reserve baseline drain rate (x-axis reference), one urgent config
./bin/urgent_baseline --mode sat --urgent-blocks 1 --trials 2000 \
  --csv results/raw/pareto/res_R0_U1.csv
for U in 1 4 16; do
  ./bin/yield --mode latency --set dev --events "$EVENTS" \
    --urgent-tiles "$U" --csv "results/raw/pareto/coop_U${U}.csv"
done
./bin/yield --mode overhead --tasks 200000 --reps 20 \
  --csv results/raw/pareto/coop_ovh.csv
echo PARETO-ALL-DONE
