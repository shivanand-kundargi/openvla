"""
train_probes.py

Train linear probes to predict concepts from language model features.

Usage:
    python -m concept_analysis.train_probes \\
        --features_path ./results/language_features_train.npz \\
        --labels_path ./results/concept_labels_train.npz \\
        --output_dir ./results/probes \\
        --layer layer_-1
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class ConceptProbe:
    """Linear probe for predicting a single concept."""
    
    def __init__(self, regularization: float = 1.0):
        """
        Initialize probe.
        
        Args:
            regularization: L2 regularization strength (inverse of C)
        """
        self.model = LogisticRegression(
            C=1.0 / regularization,
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train probe on features X and labels y."""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict concept presence (0 or 1)."""
        if not self.is_trained:
            raise ValueError("Must train probe first!")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict concept probability."""
        if not self.is_trained:
            raise ValueError("Must train probe first!")
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate probe and return metrics."""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1":  f1_score(y, y_pred, zero_division=0),
            "average_precision": average_precision_score(y, y_prob)
        }


def load_data(features_path: str, labels_path: str, layer: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load features and labels, ensuring alignment."""
    print(f"Loading features from {features_path}")
    features_data = np.load(features_path, allow_pickle=True)
    
    print(f"Loading labels from {labels_path}")
    labels_data = np.load(labels_path, allow_pickle=True)
    
    # Extract features for specified layer
    if layer not in features_data:
        available_layers = [k for k in features_data.keys() if k.startswith("layer_")]
        raise ValueError(f"Layer {layer} not found. Available: {available_layers}")
    
    X = features_data[layer]
    y = labels_data["labels"]
    vocabulary = labels_data["vocabulary"].tolist()
    
    # Check instructions alignment
    inst_features = features_data["instructions"]
    inst_labels = labels_data["instructions"]
    
    min_len = min(len(X), len(y))
    print(f"Aligning data: {len(X)} features vs {len(y)} labels -> keeping {min_len} samples")
    
    # Check if first min_len instructions align
    # We compare a small sample to avoid comparing huge arrays of strings if possible, 
    # but for correctness we should compare all. 
    # Since these are numpy arrays of objects (strings), comparison is fast enough.
    if not np.array_equal(inst_features[:min_len], inst_labels[:min_len]):
        # If strict alignment fails, try to find common indices (this handles shuffled/subset cases)
        print("Warning: Sequential alignment failed! Trying to align by matching instruction strings...")
        
        # This assumes instructions are unique identifiers (which might NOT be true for identical tasks)
        # But if they assume indices match (deterministic loading), we should just check why they differ.
        # If they differ, it usually means different splits or shuffles.
        # Let's verify if they correspond to the same dataset indices.
        # For now, let's raise an error if they don't match, because blind intersection is risky with duplicate instructions.
        
        # Check first few
        for i in range(min(5, min_len)):
             if inst_features[i] != inst_labels[i]:
                 print(f"Mismatch at index {i}:")
                 print(f"  Feature inst: {inst_features[i]}")
                 print(f"  Label inst:   {inst_labels[i]}")
        
        raise ValueError("Data misalignment! Instructions in features and labels do not match sequentially.")
    
    # Slice to common length
    X = X[:min_len]
    y = y[:min_len]
    
    print(f"\nFinal data shapes:")
    print(f"  Features: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Vocabulary size: {len(vocabulary)}")
    
    return X, y, vocabulary


def train_all_probes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    vocabulary: list,
    regularization: float = 1.0
) -> Dict[str, ConceptProbe]:
    """Train a probe for each concept."""
    num_concepts = y_train.shape[1]
    probes = {}
    results = []
    
    print(f"\nTraining {num_concepts} concept probes...")
    
    for concept_idx in tqdm(range(num_concepts)):
        concept_name = vocabulary[concept_idx]
        
        # Get labels for this concept
        y_train_c = y_train[:, concept_idx]
        y_val_c = y_val[:, concept_idx]
        
        # Skip if concept never appears
        if y_train_c.sum() == 0:
            continue
        
        # Train probe
        probe = ConceptProbe(regularization=regularization)
        probe.train(X_train, y_train_c)
        probes[concept_name] = probe
        
        # Evaluate
        metrics = probe.evaluate(X_val, y_val_c)
        results.append({
            "concept": concept_name,
            "train_pos": int(y_train_c.sum()),
            "val_pos": int(y_val_c.sum()),
            **metrics
        })
    
    return probes, results


def print_results_summary(results: list):
    """Print summary of probe training results."""
    if not results:
        print("No results to display")
        return
    
    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    
    # Compute averages
    avg_acc = np.mean([r["accuracy"] for r in results])
    avg_f1 = np.mean([r["f1"] for r in results])
    avg_ap = np.mean([r["average_precision"] for r in results])
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Average Accuracy: {avg_acc:.3f}")
    print(f"Average F1 Score: {avg_f1:.3f}")
    print(f"Average Precision: {avg_ap:.3f}")
    print(f"\nTop 10 concepts by accuracy:")
    print(f"{'-'*80}")
    print(f"{'Concept':<40} {'Accuracy':>10} {'F1':>10} {'AP':>10}")
    print(f"{'-'*80}")
    
    for result in results_sorted[:10]:
        print(f"{result['concept']:<40} "
              f"{result['accuracy']:>10.3f} "
              f"{result['f1']:>10.3f} "
              f"{result['average_precision']:>10.3f}")
    
    print(f"{'-'*80}")
    
    # Print stats by category
    categories = {}
    for r in results:
        cat = r["concept"].split(":")[0] if ":" in r["concept"] else "other"
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    print(f"\nResults by category:")
    print(f"{'-'*80}")
    print(f"{'Category':<15} {'Count':>8} {'Avg Acc':>10} {'Avg F1':>10} {'Avg AP':>10}")
    print(f"{'-'*80}")
    
    for cat in sorted(categories.keys()):
        cat_results = categories[cat]
        cat_acc = np.mean([r["accuracy"] for r in cat_results])
        cat_f1 = np.mean([r["f1"] for r in cat_results])
        cat_ap = np.mean([r["average_precision"] for r in cat_results])
        
        print(f"{cat:<15} {len(cat_results):>8} "
              f"{cat_acc:>10.3f} {cat_f1:>10.3f} {cat_ap:>10.3f}")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Train concept probes")
    parser.add_argument("--features_path", type=str, required=True,
                       help="Path to features .npz file")
    parser.add_argument("--labels_path", type=str, required=True,
                       help="Path to labels .npz file")
    parser.add_argument("--output_dir", type=str, default="./results/probes",
                       help="Output directory for trained probes")
    parser.add_argument("--layer", type=str, default="layer_-1",
                       help="Which layer to train probes on")
    parser.add_argument("--regularization", type=float, default=1.0,
                       help="L2 regularization strength")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Validation split fraction")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y, vocabulary = load_data(args.features_path, args.labels_path, args.layer)
    
    # Train/val split
    print(f"\nSplitting data: {1-args.val_split:.0%} train, {args.val_split:.0%} val")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.val_split,
        random_state=42
    )
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val: {X_val.shape[0]} samples")
    
    # Train probes
    probes, results = train_all_probes(
        X_train, y_train,
        X_val, y_val,
        vocabulary,
        regularization=args.regularization
    )
    
    # Print results
    print_results_summary(results)
    
    # Save results
    results_file = output_dir / f"probe_results_{args.layer}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to {results_file}")
    
    # Save probes (weights only to save space)
    probes_file = output_dir / f"probe_weights_{args.layer}.npz"
    probe_weights = {}
    for concept, probe in probes.items():
        probe_weights[f"{concept}_coef"] = probe.model.coef_
        probe_weights[f"{concept}_intercept"] = probe.model.intercept_
    
    np.savez_compressed(probes_file, **probe_weights)
    print(f"Saved probe weights to {probes_file}")
    
    print("\n✓ Training complete!")
    print(f"\nNext step: Analyze results and visualize concept activations")


if __name__ == "__main__":
    main()
