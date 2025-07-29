#!/usr/bin/env bash
set -euo pipefail

max_epochs_list=(150 200 2000)

for max_epochs in "${max_epochs_list[@]}"; do
    echo "=== Running with: max_epochs=$max_epochs"

    python train_double_pendulum_new.py \
        --max_epochs=${max_epochs}
done
