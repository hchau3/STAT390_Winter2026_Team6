import json
import pandas as pd
from pathlib import Path
import numpy as np

"""
This script collects data of runs in the targeted directory from json dumps and for each split: 
    - reports the minimum, average, and maximum test accuracy across runs
    - plots a boxplot of the best validation loss across runs
    - reports the percentage of correct classification for each tested case across runs
    - plots a boxplot of the predicted probability of high-grade for each tested case across runs
"""

def aggregate_results(base_directory="./"):
    """
    Recursively finds all results.json files, extracts the data, 
    saves a master JSON dump, a raw CSV table, and calculates summary statistics.
    """
    all_runs_data = []
    base_path = Path(base_directory)

    json_files = list(base_path.rglob("results.json"))
    
    if not json_files:
        print(f"No results.json files found in {base_directory}!")
        return

    print(f"Found {len(json_files)} result files. Aggregating...")

    # Gather all the data
    for file_path in json_files:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                row = {
                    "folder_name": file_path.parent.name,
                    "seed": data.get("args", {}).get("seed", "N/A"),
                    "split": data.get("args", {}).get("split", "N/A"),
                    "epochs_run": data.get("train", {}).get("epochs", None),
                    "best_val_loss": data.get("train", {}).get("best_val_loss", None),
                    "test_loss": data.get("test", {}).get("test_loss", None),
                    "test_accuracy": data.get("test", {}).get("test_accuracy", None),
                    "high_grade_recall": data.get("test", {}).get("test_high_grade_recall", None),
                    "benign_recall": data.get("test", {}).get("test_benign_recall", None)
                }
                all_runs_data.append(row)
            except json.JSONDecodeError:
                print(f"Warning: {file_path} is corrupted. Skipping.")

    # Create the Raw Summary Table
    df = pd.DataFrame(all_runs_data)
    
    # Clean up types for sorting and math (converting strings to floats/ints)
    cols_to_numeric = ['split', 'seed', 'best_val_loss', 'test_accuracy', 'test_loss']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort logic
    if "split" in df.columns and "seed" in df.columns:
        df = df.sort_values(by=['split', 'seed'])

    df.to_csv("aggregated_raw_data.csv", index=False)
    print("Saved raw data table to: aggregated_raw_data.csv")

    # Create the central JSON dump
    with open("central_results_dump.json", "w") as f:
        # We have to replace NaN with None for valid JSON serialization
        json_safe_data = df.replace({np.nan: None}).to_dict(orient="records")
        json.dump(json_safe_data, f, indent=4)
    print("Saved full JSON dump to: central_results_dump.json")

    # Calculate statistics by split
    if "split" in df.columns:
        # Group by split and calculate min, mean, max
        stats_df = df.dropna(subset=['split']).groupby('split').agg({
            'test_accuracy': ['min', 'mean', 'max'],
            'best_val_loss': ['min', 'mean', 'max']
        })
        
        # Flatten the column names (e.g., from ('test_accuracy', 'mean') to 'test_accuracy_mean')
        stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
        stats_df.reset_index(inplace=True)
        
        # Round the metrics to 4 decimal places for readability
        stats_df = stats_df.round(4)
        
        stats_df.to_csv("summary_statistics.csv", index=False)
        print("Saved summary statistics to: summary_statistics.csv\n")
        
        print("--- Quick Look: Statistics by Split ---")
        print(stats_df.to_string(index=False))

if __name__ == "__main__":
    aggregate_results(base_directory="./analyze_runs")