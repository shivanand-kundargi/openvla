"""
analyze_results.py

Analyze and visualize probe performance across network layers.
Generates plots for accuracy vs layer and concept category breakdown.

Usage:
    python -m concept_analysis.analyze_results \
        --results_dir ./concept_analysis/results/multilayer/probes \
        --output_dir ./concept_analysis/results/multilayer/analysis
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all probe result JSONs from directory."""
    files = glob.glob(os.path.join(results_dir, "probe_results_layer_*.json"))
    
    all_data = []
    
    for f in files:
        # Extract layer index from filename
        match = re.search(r"layer_(-?\d+)", f)
        if not match:
            continue
        layer_idx = int(match.group(1))
        
        # Determine logical layer number (0-32)
        # Assuming -1 is 32 (final), -33 is 0 (embedding)
        # This mapping depends on hidden_states size being 33
        # We'll use the raw index for now but label it nicely later
        
        with open(f, "r") as fp:
            results = json.load(fp)
            
        for r in results:
            # Parse concept category
            concept = r["concept"]
            category = "other"
            if ":" in concept:
                category = concept.split(":")[0]
            elif "_" in concept:
                category = concept.split("_")[0]
            
            r["layer_idx"] = layer_idx
            r["category"] = category
            all_data.append(r)
            
    return pd.DataFrame(all_data)


def plot_layer_performance(df: pd.DataFrame, output_dir: Path):
    """Plot average accuracy per layer."""
    if df.empty:
        print("No data to plot!")
        return

    # Map negative indices to approximate relative depth for sorting
    # We sort by the raw negative index ascending (-33, -25, ..., -1)
    df["sort_key"] = df["layer_idx"]
    layer_order = sorted(df["layer_idx"].unique())
    
    # 1. Overall Accuracy vs Layer
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="layer_idx", y="accuracy", marker="o", errorbar=("ci", 95))
    plt.title("Concept Probe Accuracy by Network Depth")
    plt.xlabel("Layer Index (Negative indexing: -1 is Output, -33 is Input)")
    plt.ylabel("Probe Accuracy")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "accuracy_by_layer.png", dpi=300)
    plt.close()
    
    # 2. Accuracy by Category vs Layer
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x="layer_idx", y="accuracy", hue="category", marker="o", errorbar=None)
    plt.title("Concept Accuracy by Category and Depth")
    plt.xlabel("Layer Index")
    plt.ylabel("Accuracy")
    plt.legend(title="Concept Category")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "accuracy_by_category_layer.png", dpi=300)
    plt.close()
    
    # 3. Heatmap of Top Concepts
    # Filter for top 20 most frequent concepts
    top_concepts = df.groupby("concept").count().sort_values("train_pos", ascending=False).head(20).index
    df_top = df[df["concept"].isin(top_concepts)]
    
    pivot = df_top.pivot_table(index="concept", columns="layer_idx", values="accuracy")
    # Sort columns
    pivot = pivot[sorted(pivot.columns)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Accuracy Trajectory for Top Concepts")
    plt.tight_layout()
    plt.savefig(output_dir / "top_concepts_heatmap.png", dpi=300)
    plt.close()


def analyze_action_correlations(
    features_path: str,
    labels_path: str, 
    output_dir: Path
):
    """
    Analyze correlation between:
    1. Linguistic Concepts vs Ground Truth Actions (GT)
    2. Linguistic Concepts vs Predicted Actions (Output)
    3. Ground Truth vs Predicted Actions (Alignment)
    """
    print("\nRunning Concept-Action-Prediction Correlation Analysis...")
    
    # Load data
    try:
        feat_data = np.load(features_path, allow_pickle=True)
        lbl_data = np.load(labels_path, allow_pickle=True)
    except Exception as e:
        print(f"Skipping action analysis: Could not load data ({e})")
        return

    if "actions" not in feat_data:
        print("Skipping: No 'actions' found in features file")
        return

    gt_actions = feat_data["actions"]  # [N, 7]
    labels = lbl_data["labels"]        # [N, n_concepts]
    vocab = lbl_data["vocabulary"]     # [n_concepts]
    
    has_preds = "predictions" in feat_data
    if has_preds:
        pred_actions = feat_data["predictions"] # [N, 7]
        print(f"Loaded Predictions: {pred_actions.shape}")
    else:
        print("No predictions found in features file.")
        pred_actions = None
    
    # Align lengths
    min_len = min(len(gt_actions), len(labels))
    if has_preds:
        min_len = min(min_len, len(pred_actions))
        
    gt_actions = gt_actions[:min_len]
    labels = labels[:min_len]
    if has_preds:
        pred_actions = pred_actions[:min_len]
        
    # Action dimensions (Bridge V2)
    action_dims = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    
    # --- Part 1: GT vs Pred Correlation (Accuracy check) ---
    if has_preds:
        print("\nChecking Ground Truth vs Prediction Alignment...")
        alignment_corrs = []
        for i, dim in enumerate(action_dims):
            if i >= pred_actions.shape[1]: break
            
            gt_vals = gt_actions[:, i]
            pred_vals = pred_actions[:, i]
            
            # Simple Pearson
            if np.std(gt_vals) > 1e-6 and np.std(pred_vals) > 1e-6:
                corr = np.corrcoef(gt_vals, pred_vals)[0, 1]
            else:
                corr = 0.0
            
            alignment_corrs.append({"dim": dim, "correlation": corr})
            
        df_align = pd.DataFrame(alignment_corrs)
        print(df_align.to_string())
        df_align.to_csv(output_dir / "gt_pred_alignment.csv", index=False)

    # --- Part 2: Concepts vs Actions (GT & Pred) ---
    correlations = []
    
    # Select most frequent concepts for cleaner plot
    concept_counts = labels.sum(axis=0)
    top_indices = np.argsort(concept_counts)[::-1][:30]  # Top 30 concepts
    
    for idx in top_indices:
        concept = vocab[idx]
        concept_mask = labels[:, idx] > 0
        
        if concept_mask.sum() < 10: continue
            
        # Correlate with GT
        for dim_idx, dim_name in enumerate(action_dims):
            # GT Correlation
            act_vals = gt_actions[:, dim_idx]
            if np.std(act_vals) < 1e-6:
                corr_gt = 0
            else:
                corr_gt = np.corrcoef(concept_mask, act_vals)[0, 1]
            
            entry = {
                "concept": concept,
                "action_dim": dim_name,
                "corr_gt": corr_gt,
                "count": concept_mask.sum()
            }
            
            # Pred Correlation
            if has_preds and dim_idx < pred_actions.shape[1]:
                pred_vals = pred_actions[:, dim_idx]
                if np.std(pred_vals) < 1e-6:
                    corr_pred = 0
                else:
                    corr_pred = np.corrcoef(concept_mask, pred_vals)[0, 1]
                
                entry["corr_pred"] = corr_pred
                entry["diff"] = abs(corr_gt - corr_pred)
            
            correlations.append(entry)
            
    df_corr = pd.DataFrame(correlations)
    
    if df_corr.empty:
        print("No correlations found.")
        return
        
    # Plot Heatmap (GT)
    pivot_gt = df_corr.pivot_table(index="concept", columns="action_dim", values="corr_gt")
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_gt, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Correlation: Concepts vs Ground Truth Actions")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_gt.png", dpi=300)
    plt.close()
    
    # Plot Heatmap (Pred)
    if has_preds:
        pivot_pred = df_corr.pivot_table(index="concept", columns="action_dim", values="corr_pred")
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_pred, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
        plt.title("Correlation: Concepts vs Predicted Actions")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_pred.png", dpi=300)
        plt.close()

    # Save raw data
    df_corr.to_csv(output_dir / "concept_action_correlations.csv", index=False)
    print(f"\nSaved correlation analysis to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-layer probe results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing probe_results_layer_*.json files")
    parser.add_argument("--output_dir", type=str, default="./results/analysis",
                       help="Directory to save plots")
    # Optional arguments for data paths (needed for action analysis)
    parser.add_argument("--features_path", type=str, default=None, help="Path to features npz")
    parser.add_argument("--labels_path", type=str, default=None, help="Path to labels npz")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Existing Layer Analysis
    print(f"Loading results from {args.results_dir}...")
    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} records.")
    
    if len(df) > 0:
        print("Generating plots...")
        plot_layer_performance(df, output_dir)
        print(f"Plots saved to {output_dir}")
        print("  - accuracy_by_layer.png")
        print("  - accuracy_by_category_layer.png")
        print("  - top_concepts_heatmap.png")
    else:
        print("No probe results found.")

    # 2. Action & Prediction Analysis
    if args.features_path and args.labels_path:
        analyze_action_correlations(args.features_path, args.labels_path, output_dir)
    else:
        # Try to guess paths from results_dir parent
        parent_dir = Path(args.results_dir).parent
        # Try specific layer features first or generic
        # The sequential script produces features_layer_X.npz, not language_features_train.npz
        # But run_multilayer_probe.sh moves language_features_train.npz to features_layer_X.npz
        # We need ONE feature file to get the actions/predictions. Any layer will do.
        
        candidates = list(parent_dir.glob("features_layer_*.npz"))
        lbl_path = parent_dir / "concept_labels_train.npz"
        
        if candidates and lbl_path.exists():
            feat_path = candidates[0] # Take first available layer file
            print(f"Found data files, running action/prediction analysis on {feat_path.name}...")
            analyze_action_correlations(str(feat_path), str(lbl_path), output_dir)
        else:
            print(f"Skipping action analysis: concept_labels_train.npz or features_layer_*.npz not found in {parent_dir}")

if __name__ == "__main__":
    main()
