#!/usr/bin/env bash
# Paired supervised-only vs PI-Init runs over datasets and seeds, then a summary table.
#   DATASETS="burgers advection cfd1d darcy" SEEDS="0 1 2" bash scripts/run_all.sh
# Extra arguments are passed to run.py, e.g. --budget 1000 --data-root /path/to/data
set -euo pipefail
DATASETS=${DATASETS:-"burgers advection cfd1d darcy"}
SEEDS=${SEEDS:-"0 1 2"}
OUT=${OUT:-runs}
PYTHON=${PYTHON:-python}
for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for method in supervised pi_init; do
      "$PYTHON" run.py --config "configs/${dataset}.yaml" --method "$method" --seed "$seed" --out "$OUT" "$@"
    done
  done
done
"$PYTHON" summarize.py --runs "$OUT"
