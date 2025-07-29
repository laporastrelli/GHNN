#!/usr/bin/env bash
set -euo pipefail

datasets=(
  "Data_double_pendulum" "Data_double_pendulum" "Data_double_pendulum"
)

period_q=(
  3.141592653589793 3.141592653589793 3.141592653589793
)

max_epochs=(
  150 200 2000
)

# Loop over all indices in the datasets array
for i in "${!datasets[@]}"; do
    ds="${datasets[$i]}"
    pq="${period_q[$i]}"
    ep="${max_epochs[$i]}"

    echo "=== Analyzing $ds with period_q=$pq and epochs=$ep ==="
    python analyze_pendulum.py \
        --dataset="$ds" \
        --period_q=$pq \
        --max_epochs=$ep
done
