#!/bin/bash
#SBATCH --account=e32998                              # Use your Quest project account (check with 'groups')
#SBATCH --partition=short                             # 'short', 'normal', or 'long' partition
#SBATCH --time=0:05:00                                # max run time
#SBATCH --nodes=1                                     # 1 node
#SBATCH --ntasks=1                                    # n cores
#SBATCH --mem=4G                                      # n GB RAM
#SBATCH --job-name=eval_best                          # Job name (no spaces)
#SBATCH --chdir=./                                    # set working directory
#SBATCH --output=./logs/%j_%x.out                     # Standard output file
#SBATCH --error=./logs/%j_%x.err                      # now writing error to the same log file
#SBATCH --mail-type=NONE                              # Send email on begin, end, and fail
#SBATCH --mail-user=henrychau2026@u.northwestern.edu  # Your NU email

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run training
# python3 -m pip install -r requirements.txt
# python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

# Use preset splits, given split number else 1st split, given seed else 42
python3 main.py \
    --max_slices_per_stain 5 \
    --load_splits ./no_train_val_leakage_splits/data_splits_new_0"${1:-1}".npz \
    --seed "${2:-42}" \
    --eval_only \
    --resume $3 \
    --runs_dir ./runs_best_analysis
