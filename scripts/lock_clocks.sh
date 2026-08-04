#!/usr/bin/env bash
# Step zero on any rental box, BEFORE the first measurement (ROADMAP.md:
# "If the rental denies this, that fact changes the provider, not the
# methodology"). Locks clocks, then reads them back — the read-back is the
# part that counts. Record the output in NOTEBOOK.md.
#
# Usage: ./scripts/lock_clocks.sh [SM_MHZ] [MEM_MHZ]
# A10 default is 1200, NOT the 1695 boost clock: measured 2026-08-04, the
# 150 W cap (= power.max_limit) throttles through a 1695 lock (1290 MHz
# observed) and through 1350 after ~8 s. 1200 holds with ~20 W margin.
# A lock is only evidence if the read-back is taken UNDER LOAD.
# Check supported values with: nvidia-smi -q -d SUPPORTED_CLOCKS | less
set -uo pipefail

SM_MHZ=${1:-1200}
MEM_MHZ=${2:-6251}
SUDO=$(command -v sudo >/dev/null 2>&1 && echo sudo || echo "")

echo "--- locking: persistence mode, SM ${SM_MHZ} MHz, mem ${MEM_MHZ} MHz ---"
$SUDO nvidia-smi -pm 1 || echo "WARN: persistence mode denied"
$SUDO nvidia-smi -lgc "$SM_MHZ" || echo "WARN: SM clock lock denied"
$SUDO nvidia-smi -lmc "$MEM_MHZ" || echo "WARN: mem clock lock denied"

echo "--- read-back (this is the evidence; paste into NOTEBOOK.md) ---"
nvidia-smi --query-gpu=name,persistence_mode,clocks.sm,clocks.mem,power.limit,power.max_limit --format=csv

echo "--- pass the locked SM clock to the binary: --sm-clock-mhz ${SM_MHZ} ---"
echo "If any lock was denied above, that changes the provider, not the methodology."
