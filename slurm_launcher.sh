#!/bin/bash

SEEDS='1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22'

for seed in $SEEDS; do
    export WANDB_NAME=coop_dqn_seed_${seed}
    sbatch --job-name=coop_dqn run_quentin.sh ${seed}
done
