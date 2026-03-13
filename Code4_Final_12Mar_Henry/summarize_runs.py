import os
import json
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

"""
This script collects data of runs in the targeted directory from json dumps and for each split: 
    - reports the minimum, average, and maximum test accuracy across runs
    - plots a boxplot of the best validation loss across runs
    - reports the percentage of correct classification for each tested case across runs
    - plots a boxplot of the predicted probability of high-grade for each tested case across runs
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Collect and analyze run data from json dumps")
    parser.add_argument("target_dir", type=str, help="directory containing the run folders")
    parser.add_argument("--out_dir", type=str, default="plots", help="directory to save the generated plots within target")
    return parser.parse_args()

def main():
    args = parse_args()
    
    plots_dir = os.path.join(args.target_dir, args.out_dir)
    # Ensure output directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    # Data structure: splits_data[split_number] = { metrics }
    splits_data = defaultdict(lambda: {
        'accuracies': [],
        'val_losses': [],
        'case_correct': defaultdict(list),
        'case_probs': defaultdict(list), 
        'case_true_labels': {}
    })
    
    # Collect Data
    for root, dirs, files in os.walk(args.target_dir):
        if "results.json" in files:
            folder_name = os.path.basename(root)
            # Extract split number from folder name (e.g., ..._split5_...)
            match = re.search(r'_split(\d+)_', folder_name)
            if not match:
                continue
            
            split_num = int(match.group(1))
            json_path = os.path.join(root, "results.json")
            
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                    
                    # Store run-level metrics
                    splits_data[split_num]['val_losses'].append(data['train']['best_val_loss'])
                    splits_data[split_num]['accuracies'].append(data['test']['test_accuracy'])
                    
                    # Store case-level metrics
                    preds = data['test']['predictions']
                    trues = data['test']['true_labels']
                    cases = data['test']['case_ids']
                    probs = data['test']['prediction_probs']
                    
                    for pred, true, case_id, prob in zip(preds, trues, cases, probs):
                        # Track if the prediction matched the true label
                        splits_data[split_num]['case_correct'][case_id].append(pred == true)

                        # Store the true label for this case
                        splits_data[split_num]['case_true_labels'][case_id] = true
                        
                        # Stores predicted probability of class 1
                        splits_data[split_num]['case_probs'][case_id].append(prob[1])
                        
                except KeyError as e:
                    print(f"Warning: Missing expected key {e} in {json_path}. Skipping file.")
                except Exception as e:
                    print(f"Warning: Failed to parse {json_path}: {e}")

    if not splits_data:
        print("No valid results.json files found matching the folder pattern '<unique_id>_split<n>_seed<x>'.")
        return

    # Process and Report per Split
    sns.set_theme(style="whitegrid")

    # List to collect validation losses for the combined plot
    combined_val_loss_data = []
    
    for split_num in sorted(splits_data.keys()):
        print(f"\n{'='*40}")
        print(f" SPLIT {split_num} REPORT")
        print(f"{'='*40}")
        
        data = splits_data[split_num]
        
        # --- Accuracy Reporting ---
        accs = data['accuracies']
        if accs:
            print(f"Test Accuracy across {len(accs)} runs:")
            print(f"  Min: {np.min(accs):.4f} | Avg: {np.mean(accs):.4f} | Max: {np.max(accs):.4f}\n")
            
        # --- Case Classification Reporting ---
        print("Correct Classification Percentage per Case:")
        # Sort cases numerically if possible, otherwise as strings
        sorted_cases = sorted(data['case_correct'].keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        
        for case_id in sorted_cases:
            correct_list = data['case_correct'][case_id]
            pct_correct = (sum(correct_list) / len(correct_list)) * 100
            print(f"  Case {case_id:<5}: {pct_correct:>6.2f}%  ({sum(correct_list)}/{len(correct_list)} runs)")

        # --- Collect data for Combined Validation Loss Plot ---
        for loss in data['val_losses']:
            combined_val_loss_data.append({
                "Split": f"Split {split_num}",
                "Validation Loss": loss
            })

        # --- Plotting: Predicted Probability per Case ---
        plot_data = []
        for case_id in sorted_cases:
            true_label = data['case_true_labels'][case_id]
            # Map numerical labels to readable strings for the legend
            label_name = "High-Grade (1)" if true_label == 1 else "Benign (0)"

            for prob in data['case_probs'][case_id]:
                plot_data.append({
                    "Case ID": str(case_id), 
                    "Probability of High-Grade": prob, 
                    "True Class": label_name
                })
                
        df_probs = pd.DataFrame(plot_data)
        
        plt.figure(figsize=(max(8, len(sorted_cases) * 0.4), 6)) # Dynamically scale width based on cases
        sns.boxplot(
            data=df_probs, 
            x="Probability of High-Grade", 
            y="Case ID", 
            palette={"High-Grade (1)": "salmon", "Benign (0)": "lightgreen"}, 
            hue="True Class"
        )
        plt.title(f"Predicted Probability of High-Grade per Tested Case - Split {split_num}")
        plt.xlabel("Case ID")
        plt.ylabel("Probability (Class 1)")
        plt.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Decision Threshold')
        plt.xticks(rotation=45)
        plt.legend(title="True Class", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        case_probs_path = os.path.join(plots_dir, f"split_{split_num}_case_probs.png")
        plt.savefig(case_probs_path)
        plt.close()
        print(f"\n-> Saved Case Probability boxplot to: {case_probs_path}")

    # --- Plotting: Best Validation Loss ---
    df_val_loss = pd.DataFrame(combined_val_loss_data)
    
    plt.figure(figsize=(max(6, len(splits_data) * 1.5), 5))
    sns.boxplot(data=df_val_loss, x="Split", y="Validation Loss", color="skyblue")
    plt.title("Best Validation Loss Across Runs by Split")
    plt.xlabel("Split")
    plt.ylabel("Validation Loss")
    plt.tight_layout()
    
    combined_val_loss_path = os.path.join(plots_dir, "combined_val_loss.png")
    plt.savefig(combined_val_loss_path)
    plt.close()
    print(f"\n{'='*40}")
    print(f"-> Saved Combined Validation Loss boxplot to: {combined_val_loss_path}")

if __name__ == "__main__":
    main()