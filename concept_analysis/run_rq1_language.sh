#!/bin/bash
# Quick start script for RQ1: Language Encoder Concept Probing

set -e

echo "========================================="
echo "RQ1: Language Encoder Concept Probing"
echo "========================================="
echo ""

# Configuration
DATA_ROOT="${DATA_ROOT:-./concept_analysis/data}"
OUTPUT_DIR="${OUTPUT_DIR:-./concept_analysis/results}"
MODEL_NAME="${MODEL_NAME:-openvla/openvla-7b}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"  # Use subset for faster testing

# Check if data root exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: DATA_ROOT not found: $DATA_ROOT"
    echo "Please set DATA_ROOT environment variable to your bridge_orig path"
    echo "Example: export DATA_ROOT=/path/to/bridge_orig"
    exit 1
fi

# Install dependencies
# echo "Step 0: Installing dependencies..."
# echo "---------------------------------"
# pip install -q -r concept_analysis/requirements.txt
# python -m spacy download en_core_web_sm
# echo "✓ Dependencies installed"
# echo ""

# # Step 1: Extract concepts
# echo "Step 1: Extracting concepts from instructions..."
# echo "---------------------------------"
# python -m concept_analysis.extract_concepts \
#     --data_root "$DATA_ROOT" \
#     --output_dir "$OUTPUT_DIR" \
#     --min_frequency 10 \
#     --split train

# echo "✓ Concepts extracted"
# echo ""

# Step 2: Extract language features
echo "Step 2: Extracting language features from OpenVLA..."
echo "---------------------------------"
python -m concept_analysis.extract_features \
    --model_name "$MODEL_NAME" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --split train \
    --batch_size 8 \
    --max_samples "$MAX_SAMPLES" \
    --layers "-1"

echo "✓ Features extracted"
echo ""

# Step 3: Train probes
echo "Step 3: Training linear concept probes..."
echo "---------------------------------"
python -m concept_analysis.train_probes \
    --features_path "$OUTPUT_DIR/language_features_train.npz" \
    --labels_path "$OUTPUT_DIR/concept_labels_train.npz" \
    --output_dir "$OUTPUT_DIR/probes" \
    --layer "layer_-1" \
    --regularization 1.0 \
    --val_split 0.2

echo "✓ Probes trained"
echo ""

echo "========================================="
echo "RQ1 Complete!"
echo "========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files created:"
echo "  - $OUTPUT_DIR/concept_vocabulary.txt"
echo "  - $OUTPUT_DIR/concept_labels_train.npz"
echo "  - $OUTPUT_DIR/language_features_train.npz"
echo "  - $OUTPUT_DIR/probes/probe_results_layer_-1.json"
echo ""
echo "Next steps:"
echo "  1. Review results: cat $OUTPUT_DIR/probes/probe_results_layer_-1.json | jq '.[:10]'"
echo "  2. Run on test set for final evaluation"
echo "  3. Try different layers: --layers '-3,-2,-1'"
echo ""
