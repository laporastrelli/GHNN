#!/usr/bin/env bash
set -euo pipefail

# Fixed argument
NUM_RUNS=10

# The values you want to sweep:
# (omit a flag by setting it to an empty string)
HALF_FLAGS=( 0 1 )
INIT_OPTS=( rand )
horizon=( 25 50 100 )

# Loop over every combination:
for half_arg in "${HALF_FLAGS[@]}"; do
  for init_arg in "${INIT_OPTS[@]}"; do
    for horizon_arg in "${horizon[@]}"; do
      echo "=== Running with: --half $half_arg --init $init_arg --num_runs $NUM_RUNS --horizon $horizon_arg ==="

      python generate_single_pendulum_dataset.py \
          --init="$init_arg" \
          --half=$half_arg \
          --horizon=$horizon_arg \
          --num_runs=$NUM_RUNS 
    done
  done
done
