#!/bin/bash
# Quick start script for RQ1 with auto-download (no manual dataset needed)

set -e

echo "========================================="
echo "RQ1: Language Encoder Concept Probing"
echo "Using TensorFlow Datasets auto-download"
echo "========================================="
echo ""

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-./concept_analysis/results}"
MODEL_NAME="${MODEL_NAME:-openvla/openvla-7b}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"  # Use subset for faster testing

echo "Note: First run will download Bridge V2 (~124GB) to ~/tensorflow_datasets/"
echo "This may take a while depending on your internet connection."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Install dependencies
echo "Step 0: Installing dependencies..."
echo "---------------------------------"
pip install -q -r concept_analysis/requirements.txt
python -m spacy download en_core_web_sm
echo "✓ Dependencies installed"
echo ""

# Step 1: Extract concepts (auto-download dataset)
echo "Step 1: Extracting concepts from instructions..."
echo "---------------------------------"
python -m concept_analysis.extract_concepts \
    --output_dir "$OUTPUT_DIR" \
    --min_frequency 10 \
    --split train

echo "✓ Concepts extracted"
echo ""

# Step 2: Extract language features (auto-download dataset)
echo "Step 2: Extracting language features from OpenVLA..."
echo "---------------------------------"
python -m concept_analysis.extract_features \
    --model_name "$MODEL_NAME" \
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
echo "Dataset cached at: ~/tensorflow_datasets/bridge_dataset/"
echo ""
