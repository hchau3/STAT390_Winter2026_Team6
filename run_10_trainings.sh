#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Seeds to pass for each run
seeds=(1 4 9 16 25 36 49 64 81 100)
# Define splits to iterate through
splits=(1 2 3 4 5)

for seed in "${seeds[@]}"; do
    for split in "${splits[@]}"; do
        echo "Submitting: Split $split, Seed $seed"
        
        sbatch --job-name="split${split}_seed${seed}" \
            "sbatch_files/run_training_split${split}.sbatch" \
            "$seed"
    done
done
