# Hierarchical Attention MIL Trainer

## Overview

This implementation converts the original Jupyter notebook into a modular Python structure suitable for Quest. The model uses three levels of attention:

1. **Patch-level attention**: Within each stain-slice
2. **Stain-level attention**: Across slices within each stain  
3. **Case-level attention**: Across different stains (H&E, Melan, SOX10)


## Usage

### Training with Attention Analysis
```bash
python main.py --analyze_attention --attention_top_n 5
```

### Evaluation Only
```bash
python main.py --eval_only --resume /path/to/checkpoint.pth --analyze_attention
```

### Using Existing Data Splits
```bash
python main.py --load_splits ./runs/run_20241028_143022/data_splits.npz
```

### Advanced Options
```bash
python main.py \
    --labels_csv /path/to/labels.csv \
    --patches_dir /path/to/patches \
    --runs_dir /path/to/runs \
    --epochs 10 \
    --lr 1e-4 \
    --batch_size 32 \
    --num_workers 2 \
    --embed_dim 512 \
    --per_slice_cap 800 \
    --max_slices_per_stain 5 \
    --seed 42 \
    --resume /path/to/checkpoint.pth \
    --eval_only \
    --analyze_attention \
    --attention_top_n 10 \
    --load_splits /path/to/data_splits.npz
```

**Key Arguments:**
You can specify any combination of these arguments to override the defaults in `config.py`:
- `--labels_csv`: Path to labels CSV file
- `--patches_dir`: Path to patches directory
- `--runs_dir`: Path to runs directory
- `--epochs`: Number of training epochs (default: 30)
- `--lr`: Learning rate (default: 3e-4)
- `--batch_size`: Batch size (default: 1)
- `--num_workers`: Number of data loader workers (default: 2)
- `--embed_dim`: Embedding dimension (default: 512)
- `--per_slice_cap`: Max patches per slice (default: 800)
- `--max_slices_per_stain`: Max slices per stain (default: None)
- `--seed`: Random seed (default: 42)
- `--resume`: Resume from checkpoint
- `--eval_only`: Skip training, only evaluate
- `--analyze_attention`: Enable attention analysis and visualization
- `--attention_top_n`: Number of top/bottom patches to visualize (default: 5)
- `--load_splits`: Load existing train/val/test splits from .npz file

**Additional Config Settings:**
You can additionally set configurations through `config.py`, some of which are not available from the CLI:
- `DATA_PATHS`: Collection of relevant paths the script should know
- `MODEL_CONFIG`: Hierarchical Attention MIL setup and bag of instances specifications
- `TRAINING_CONFIG`: Training running logic and hyperparameters
- `SPLIT_CONFIG`: Specify ratios for train/val/test sets, stratify config is not used as we always stratify
- `GROUPED_CASE`: List cases that need to grouped together while splitting, used for pseudo-cases
- `IMAGE_CONFIG`: Image processing and transformations
- `VALID_CLASSES`: Used case with selected class (0: invalid, 1: benign, 2: low-grade, 3: high-grade, 4: MIS)
- `DEVICE`: Specify using GPU if available else CPU

**Output Directory:**
The output directory is automatically created with a timestamp in the format `.../YYYYMMDD_HHMMSS_<JOB_NAME>/` under a base directory, which can be changed by modifying `DATA_PATHS['runs_dir']` in `config.py` (currently set to `./runs`).

## SLURM Job Management

### Submitting Jobs

**Before submitting:**
1. Update the email in the sbatch file: Replace `YOUR_EMAIL@u.northwestern.edu` with your actual email
2. Verify the account and partition settings match your Quest allocation
3. Modify logging directory if needed: Logs are currently set to `./logs/` in the sbatch files

The `sbatch_files/` directory contains pre-configured SLURM scripts for running training with different non-overlapping data splits:

```bash
# Submissions should be made from the code base directory (the folder containing this README)
sbatch sbatch_files/run_preset_splits.sbatch <split> <seed>

# Submit a single job for one split
sbatch sbatch_files/run_preset_splits.sbatch # defaults to 1st split and seed 42

# Or submit all 5 splits for cross-validated results
sbatch sbatch_files/run_preset_splits.sbatch 1
sbatch sbatch_files/run_preset_splits.sbatch 2
sbatch sbatch_files/run_preset_splits.sbatch 3
sbatch sbatch_files/run_preset_splits.sbatch 4
sbatch sbatch_files/run_preset_splits.sbatch 5
```

**Training Strategy:**
- Use **one split** for initial experimentation or single model training
- Use **all 5 splits** for robust cross-validated results and performance evaluation

When you submit a job, SLURM will return a job ID (e.g., "Submitted batch job 122345"). You can also get job IDs later using `squeue --me`.

### Monitoring Jobs
```bash
# Check job status
squeue --me

# View real-time log output (replace <JOB_ID> with actual job ID from submission)
tail -f ./logs/<JOB_ID>_preset_split.log

# Cancel a job if needed (replace <JOB_ID> with actual job ID)
scancel <JOB_ID>
```

**Note:** Replace `<JOB_ID>` with the actual job ID returned by SLURM when you submit (e.g., if SLURM says "Submitted batch job 122345", use 122345 as the job ID).

### Job Output
- **SLURM logs** are written to `./logs/<JOB_ID>_preset_split.log` (configured in sbatch files)
- **Training results** are saved in `./runs/YYYYMMDD_HHMMSS_preset_split/` (configured in config.py)

## Output Structure

Each run creates a timestamped directory with all results:

```
./runs/YYYYMMDD_HHMMSS_<JOB_NAME>/
├── attention_analysis/            # (if --analyze_attention)
│    ├── attention_summary.txt
│    ├── top_effective_patches_per_case_5.0pct.csv
│    ├── top_effective_patches_per_case_summary_5.0pct.txt
│    ├── case_effective_patches/
│    │   ├── case_*_bottom_effective_patches.png
│    │   └── case_*_top_effective_patches.png
│    ├── patch_attention/
│    │   ├── case_*_*_slice*_bottom_patches.png
│    │   └── case_*_*_slice*_top_patches.png
│    ├── plot/
│    │   └── effective_patch_attn_distro_case_*.png
│    └── slice_attention/
│        └── slice_attn_rankplot_case_*.png
├── checkpoints/                   # Model weights
│    ├── *.pth                     # If not --eval_only
│    └── best.pth                  # If not --eval_only, model at epoch with lowest val loss
├── confusion_matrix.png           # Visual confusion matrix
├── data_splits.npz                # Case IDs for reproducibility
├── deviation_plot.png             # True Label vs Predicted Probability
├── model_loss.png                 # Training and Validation Loss graph
├── predictions.csv                # Per-case predictions
├── results.json                   # Summary metrics (for script use)
└── results.txt                    # Summary metrics
```

## Output Files

### Always Generated

**results.txt**
- Best validation loss and final epoch (if not --eval_only)
- Test loss, accuracy, high-grade recall, and low-grade recall
- Number of samples
- Checkpoint information

**results.json**
- script parseable version of results.txt

**predictions.csv**
- `case_id`: Case identifier
- `true_label`: Ground truth (0=benign, 1=high-grade)
- `predicted_label`: Model prediction
- `prob_benign`: Probability for benign class
- `prob_high_grade`: Probability for high-grade class
- `correct`: Boolean indicating correct prediction

**confusion_matrix.png**
- confusion matrix with counts for predictions on the test set

**deviation_plot.png**
- visual representation of true label versus predicted probability for cases in the test set

**model_loss.png**
- training and validation loss versus epoch history

**data_splits.npz**
Contains case IDs for each split:
- `train_cases`: Training set case IDs
- `val_cases`: Validation set case IDs
- `test_cases`: Test set case IDs

### Generated Only During Training
- checkpoint files

### Optional: Attention Analysis (--analyze_attention)

**attention_summary.txt**
- Most attended stain per case
- Stain-level attention weights
- Slice-level attention patterns

**patch_attention/ folder**
- Top N most attended patches per slice (highest/lowest slice-level attention)
- Bottom N least attended patches per slice
- Images (after transformation) with attention weights

**case_effective_patches/ folder**
- Top N patches across entire case using effective attention (stain × slice × patch weights)
- Bottom N patches across entire case using effective attention
- Provides global view of most important patches per case

**plots/ folder**
- `effective_patch_attn_distro_case_*.png`: Per-case histograms of effective patch attention
- Shows distribution and concentration of attention within each case

**slice_attention/ folder**
- `slice_attn_rankplot_case_*.png`: Per-case slice attention rankings by stain
- Bar plots showing which slices get most attention within each stain
- Includes uniform attention reference line

**Analysis CSVs and Summaries**
- `top_effective_patches_per_case_5.0pct.csv`: Detailed data on top 5% patches per case
- `top_effective_patches_per_case_summary_5.0pct.txt`: Human-readable summary of top patches
- Includes stain distribution and slice coverage statistics

## Data Format

The model expects:
- **Patches**: PNG images organized by case and stain
- **Labels CSV**: Case IDs with corresponding class labels
- **Naming Convention**: `case_{case_id}_{slice_id}_{stain}_patch{n}.png`

## Project Structure

```
Code4_Final_12_Mar/
├── attention_analysis.py               # Attention visualization
├── case_grade_match.csv                # Original data
├── check.py                            # Code testing
├── config.py                           # Configuration and paths
├── data_utils.py                       # Data loading and preprocessing
├── dataset_images.py                   # Dataset classes and transforms (images)
├── dataset.py                          # Dataset classes and transforms (embeddings)
├── extra_benign_case_grade_match.csv   # Data after augmentation
├── KimiaNet.pth                        # Pre-trained weights from KimiaNet
├── main_partial.py                     # Code testing
├── main.py                             # Main training script
├── make_splits.py                      # Create the premade splits in data_splits/
├── models.py                           # Model architectures
├── optuna_training.py                  # Base optuna script
├── precompute_pooled_features.py       # Process images and save embeddings
├── requirements.txt                    # Dependencies
├── run_10_trainings.sh                 # Bulk submit slurm jobs
├── summarize_runs.py                   # Reads results.json and create summary
├── trainer.py                          # Training and validation logic
├── utils.py                            # Helper functions
├── visualize_predictions.ipynb         # Make various plots and tables
├── no_train_val_leakage_splits/        # Pre-generated train/val/test splits
│   ├── data_splits_new_01.npz
│   ├── data_splits_new_02.npz
│   ├── data_splits_new_03.npz
│   ├── data_splits_new_04.npz
│   └── data_splits_new_05.npz
└── sbatch_files/                       # SLURM job scripts
    ├── check_partial_main.sbatch       # Code testing
    ├── optuna_sbatch.sbatch            # Runs optuna tuning script
    ├── precompute_embeddings.sbatch    # Runs image to embedding script
    ├── run_eval_best.sbatch            # Evalates model from epoch with lowest val loss
    ├── run_preset_split.sbatch         # Generic script for running preset 1-5 splits
    ├── run_splits_10x.sbatch           # Generic script running without attention_analysis
    ├── split_script.sbatch             # Runs data splitting script
    └── summarize_runs.sbatch           # Runs summarize runs script
```