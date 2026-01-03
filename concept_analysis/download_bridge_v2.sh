#!/bin/bash
# Download Bridge V2 dataset from Berkeley
# Dataset size: ~124 GB
# This is the original version from the Berkeley website (not the outdated Open-X version)

set -e

echo "========================================="
echo "Bridge V2 Dataset Downloader"
echo "========================================="
echo ""

# Configuration
DEFAULT_DATA_DIR="$HOME/datasets"
DATA_DIR="${1:-$DEFAULT_DATA_DIR}"
DATASET_URL="https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/"
DATASET_SIZE_GB=124

echo "Download destination: $DATA_DIR"
echo "Dataset size: ~${DATASET_SIZE_GB} GB"
echo ""

# Check if data directory exists, create if not
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

# Check available disk space
echo "Checking available disk space..."
AVAILABLE_SPACE_KB=$(df -k "$DATA_DIR" | tail -1 | awk '{print $4}')
AVAILABLE_SPACE_GB=$((AVAILABLE_SPACE_KB / 1024 / 1024))
REQUIRED_SPACE_GB=$((DATASET_SIZE_GB + 10))  # Add 10GB buffer

echo "Available space: ${AVAILABLE_SPACE_GB} GB"
echo "Required space: ${REQUIRED_SPACE_GB} GB (including buffer)"
echo ""

if [ $AVAILABLE_SPACE_GB -lt $REQUIRED_SPACE_GB ]; then
    echo "❌ ERROR: Not enough disk space!"
    echo "   Available: ${AVAILABLE_SPACE_GB} GB"
    echo "   Required: ${REQUIRED_SPACE_GB} GB"
    echo ""
    echo "Please free up space or choose a different directory:"
    echo "   $0 /path/to/directory/with/more/space"
    exit 1
fi

echo "✓ Sufficient disk space available"
echo ""

# Check if dataset already exists
if [ -d "$DATA_DIR/bridge_orig" ]; then
    echo "⚠️  WARNING: bridge_orig directory already exists!"
    echo "   Location: $DATA_DIR/bridge_orig"
    echo ""
    read -p "Do you want to re-download and overwrite? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Download cancelled."
        echo "Existing dataset location: $DATA_DIR/bridge_orig"
        exit 0
    fi
    echo "Removing existing dataset..."
    rm -rf "$DATA_DIR/bridge_orig"
fi

if [ -d "$DATA_DIR/bridge_dataset" ]; then
    echo "⚠️  WARNING: bridge_dataset directory already exists!"
    echo "   This might be from a previous incomplete download."
    echo ""
    read -p "Remove and start fresh? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$DATA_DIR/bridge_dataset"
    fi
fi

# Confirm download
echo "Ready to download Bridge V2 dataset"
echo ""
echo "Details:"
echo "  Source: $DATASET_URL"
echo "  Destination: $DATA_DIR"
echo "  Size: ~${DATASET_SIZE_GB} GB"
echo "  Time estimate: 30-60 minutes (depends on connection)"
echo ""
read -p "Continue with download? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

echo ""
echo "Starting download..."
echo "========================================="
echo ""

# Change to data directory
cd "$DATA_DIR"

# Download with wget
# -r: recursive
# -nH: no host directories
# --cut-dirs=4: skip 4 directory levels
# --reject="index.html*": don't download index files
# -nc: no-clobber (don't re-download existing files)
# --progress=bar:force: show progress bar
# -e robots=off: ignore robots.txt

wget -r -nH --cut-dirs=4 \
     --reject="index.html*" \
     -nc \
     --progress=bar:force \
     -e robots=off \
     "$DATASET_URL"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Download failed!"
    echo "   You can resume by running this script again."
    echo "   wget will resume from where it left off."
    exit 1
fi

echo ""
echo "========================================="
echo "Download complete!"
echo "========================================="
echo ""

# Rename to bridge_orig (required by OpenVLA)
if [ -d "bridge_dataset" ]; then
    echo "Renaming bridge_dataset -> bridge_orig..."
    mv bridge_dataset bridge_orig
    echo "✓ Renamed to bridge_orig"
else
    echo "⚠️  WARNING: bridge_dataset directory not found after download"
    echo "   Expected location: $DATA_DIR/bridge_dataset"
    ls -la "$DATA_DIR/" | grep -i bridge
    exit 1
fi

echo ""
echo "========================================="
echo "Setup Complete! ✓"
echo "========================================="
echo ""
echo "Dataset location: $DATA_DIR/bridge_orig"
echo ""

# Verify dataset structure
echo "Verifying dataset structure..."
if [ -d "$DATA_DIR/bridge_orig" ]; then
    NUM_FILES=$(find "$DATA_DIR/bridge_orig" -type f | wc -l)
    DATASET_SIZE=$(du -sh "$DATA_DIR/bridge_orig" | cut -f1)
    echo "  Files found: $NUM_FILES"
    echo "  Total size: $DATASET_SIZE"
    echo ""
    
    # Check for expected subdirectories
    if [ -d "$DATA_DIR/bridge_orig/1.0.0" ] || [ -d "$DATA_DIR/bridge_orig/0.1.0" ]; then
        echo "✓ Dataset structure looks correct"
    else
        echo "⚠️  Warning: Expected version directory not found"
        echo "   Contents of bridge_orig:"
        ls -la "$DATA_DIR/bridge_orig/"
    fi
else
    echo "❌ ERROR: bridge_orig directory not found!"
    exit 1
fi

echo ""
echo "========================================="
echo "Next Steps"
echo "========================================="
echo ""
echo "1. Use this dataset with OpenVLA concept analysis:"
echo ""
echo "   export DATA_ROOT=$DATA_DIR"
echo ""
echo "2. Run concept extraction:"
echo ""
echo "   cd /umbc/rs/pi_gokhale/users/shivank2/shivanand/openvla"
echo "   python -m concept_analysis.extract_concepts \\"
echo "       --data_root \$DATA_ROOT \\"
echo "       --output_dir ./concept_analysis/results"
echo ""
echo "3. Or run the full pipeline:"
echo ""
echo "   bash concept_analysis/run_rq1_language.sh"
echo ""
