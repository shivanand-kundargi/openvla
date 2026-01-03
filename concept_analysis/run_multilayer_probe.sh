#!/bin/bash
# Run multi-layer probing analysis for RQ1
set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-./concept_analysis/results/multilayer}"
MODEL_NAME="${MODEL_NAME:-openvla/openvla-7b}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
# Don't set a default DATA_ROOT that points to an empty folder, 
# otherwise the script tries to load from there and fails.
# Leave it empty to trigger auto-download.
DATA_ROOT="${DATA_ROOT:-}" 

# Define layers to probe (Llama-2-7b has 32 layers, hidden_states has 33 elements: 0=embed, 1-32=layers)
# 32 = -1 (Final output)
# 24 = -9 (Late)
# 16 = -17 (Middle)
# 8  = -25 (Early)
# 0  = -33 (Embedding)
LAYERS_LIST="32,24,16,8,0"

echo "========================================="
echo "RQ1: Multi-Layer Concept Probing"
echo "Layers: $LAYERS_LIST"
echo "========================================="
echo ""

mkdir -p "$OUTPUT_DIR"

# Configure TFDS to use local project directory (avoid small home drive)
# We set this globally so both 'install' and 'run' use it
export TFDS_DATA_DIR="$(pwd)/concept_analysis/data/bridge_dataset/1.0.0/"
export TMPDIR="$(pwd)/concept_analysis/data/tmp"
mkdir -p "$TFDS_DATA_DIR" "$TMPDIR"
echo "TFDS Data Directory: $TFDS_DATA_DIR"

# Step 0: Ensure Data is Converted to Images (The "Image Pipeline")
METADATA_FILE="./concept_analysis/data/images/metadata.json"

if [ ! -f "$METADATA_FILE" ]; then
    echo "========================================="
    echo "Step 0: Converting TFRecords to Images..."
    echo "========================================="
    
    # We run the conversion script. It handles the TFDS loading.
    python concept_analysis/convert_bridge_to_images.py \
        --max_samples "$MAX_SAMPLES" \
        ${DATA_ROOT:+--data_root "$DATA_ROOT"}
        
    if [ ! -f "$METADATA_FILE" ]; then
        echo "Error: Image conversion failed. Metadata file not found."
        exit 1
    fi
else
    echo "Step 0: Images already converted (metadata found)."
fi

# Step 1: Extract Concepts from Instructions
# Note: extract_concepts.py now automatically looks for the metadata.json we just verified
if [ ! -f "$OUTPUT_DIR/concept_labels_train.npz" ]; then
    echo ""
    echo "========================================="
    echo "Step 1: Extracting linguistic concepts..."
    echo "========================================="
    python -m concept_analysis.extract_concepts \
        --output_dir "$OUTPUT_DIR" \
        --min_frequency 10
else
    echo "Step 1: Concepts already extracted."
fi

# Step 2 & 3: Extract and Train Per Layer (Sequential Loop)
echo ""
echo "Step 2 & 3: Processing layers sequentially..."

# Convert comma-separated string to array
IFS=',' read -ra LAYERS <<< "$LAYERS_LIST"

for layer_idx in "${LAYERS[@]}"; do
    layer_name="layer_${layer_idx}"
    echo "========================================="
    echo "Processing $layer_name..."
    echo "========================================="
    
    # 2a. Extract Features for single layer
    echo "Extracting features..."
    python -m concept_analysis.extract_features \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        ${DATA_ROOT:+--data_root "$DATA_ROOT"} \
        --split train \
        --batch_size 8 \
        --max_samples "$MAX_SAMPLES" \
        --layers "$layer_idx"
        
    # User the file created (rename to keep it safe)
    mv "$OUTPUT_DIR/language_features_train.npz" "$OUTPUT_DIR/features_${layer_name}.npz"
    
    # 2b. Train Probe
    echo "Training probe..."
    python -m concept_analysis.train_probes \
        --features_path "$OUTPUT_DIR/features_${layer_name}.npz" \
        --labels_path "$OUTPUT_DIR/concept_labels_train.npz" \
        --output_dir "$OUTPUT_DIR/probes" \
        --layer "$layer_name" \
        --regularization 1.0 \
        --val_split 0.2
        
    echo "✓ $layer_name complete"
done

# Step 4: Analyze Results
echo ""
echo "Step 4: Analyzing and visualizing results..."
python -m concept_analysis.analyze_results \
    --results_dir "$OUTPUT_DIR/probes" \
    --output_dir "$OUTPUT_DIR/analysis"

echo ""
echo "========================================="
echo "Multi-Layer Analysis Complete!"
echo "========================================="
echo "Results saved in: $OUTPUT_DIR/probes"
echo "Plots saved in:   $OUTPUT_DIR/analysis"
