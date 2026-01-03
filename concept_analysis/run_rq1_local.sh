#!/bin/bash
# Run RQ1 with local dataset

set -e

echo "========================================="
echo "RQ1: Language Encoder Concept Probing"
echo "Using local dataset: ./concept_analysis/data"
echo "========================================="
echo ""

# Configuration
DATA_ROOT="./concept_analysis/data"
OUTPUT_DIR="./concept_analysis/results"
MODEL_NAME="${MODEL_NAME:-openvla/openvla-7b}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}" 

# Step 1: Extract concepts
echo "Step 1: Extracting concepts..."
python -m concept_analysis.extract_concepts \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --min_frequency 10

# Step 2: Extract features
echo "Step 2: Extracting features..."
python -m concept_analysis.extract_features \
    --model_name "$MODEL_NAME" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples "$MAX_SAMPLES"

# Step 3: Train probes
echo "Step 3: Training probes..."
python -m concept_analysis.train_probes \
    --features_path "$OUTPUT_DIR/language_features_train.npz" \
    --labels_path "$OUTPUT_DIR/concept_labels_train.npz" \
    --output_dir "$OUTPUT_DIR/probes"

echo "✓ Done! Results in $OUTPUT_DIR"
