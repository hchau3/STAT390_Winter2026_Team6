#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# get filepaths for best checkpoint
readarray -t bests < <(find runs_seed_analysis/ -path '*best.pth')

for best in "${bests[@]}"; do
	split=$(echo "$best" | grep -oP "split\K[0-9]+")
    seed=$(echo "$best" | grep -oP "seed\K[0-9]+")
    echo "Submitting: Split $split, Seed $seed, Checkpoint $best" 
    sbatch --job-name="split${split}_seed${seed}" \
        "sbatch_files/run_eval_best.sbatch" \
        "$split" "$seed" "$best"
done
