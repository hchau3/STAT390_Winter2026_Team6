#!/bin/bash
#SBATCH --account=e32998                              # Use your Quest project account (check with 'groups')
#SBATCH --partition=long                              # 'short', 'normal', or 'long' partition
#SBATCH --time=04:00:00                               # max run time
#SBATCH --nodes=1                                     # 1 node
#SBATCH --ntasks=1                                    # n cores
#SBATCH --mem=32G                                     # n GB RAM
#SBATCH --array=0-49
#SBATCH --job-name=MIL_Task2                          # Job name (no spaces)
#SBATCH --chdir=./                                    # set working directory
#SBATCH --output=./logs/%A_%a.out                    
#SBATCH --error=./logs/%A_%a.err
#SBATCH --mail-type=ALL                               # Send email on begin, end, and fail
#SBATCH --mail-user=vickyxu2026@u.northwestern.edu  # Your NU email

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Run training
# python3 -m pip install -r requirements.txt
# python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

# 1. compute Split and Seed 
SPLIT_NUM=$(( (SLURM_ARRAY_TASK_ID / 10) + 1 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 10 ))

# 2. define 10 seed
SEEDS=(10 20 30 40 50 60 70 80 90 100)
CURRENT_SEED=${SEEDS[$SEED_IDX]}
SPLIT_FILE="./no_train_val_leakage_splits/data_splits_new_0${SPLIT_NUM}.npz"

echo "------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running $SPLIT_FILE with Seed $CURRENT_SEED"
echo "Start time: $(date)"
echo "------------------------------------------------"

# 3. run task2 command
python3 main.py \
    --load_splits "$SPLIT_FILE" \
    --seed "$CURRENT_SEED" \
    --epochs 30

echo "------------------------------------------------"
echo "End time: $(date)"
echo "------------------------------------------------"


