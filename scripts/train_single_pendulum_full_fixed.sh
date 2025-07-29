#!/usr/bin/env bash
set -euo pipefail

period_q_list=(3.141592653589793 0)
max_epochs_list=(100 2000)
dataset='Data_full_fixed_extrapolation'

for period_q in "${period_q_list[@]}"; do
    for max_epochs in "${max_epochs_list[@]}"; do
        echo "=== Running with: period_q=$period_q, max_epochs=$max_epochs"

        python train_pendulum.py \
            --dataset=${dataset} \
            --period_q=${period_q} \
            --max_epochs=${max_epochs}
    done
done