# RQ1 Language Encoder Implementation - Quick Start Guide

## Overview

Implementation of **RQ1: Language Encoder Concept Probing** - discovering which linguistic concepts OpenVLA's language model activates when processing task instructions.

## What We've Built

### Core Scripts

1. **`extract_concepts.py`** - Extracts linguistic concepts from instructions
   - Parses nouns (objects), verbs (actions), adjectives (attributes)
   - Identifies spatial relations and action primitives
   - Creates concept vocabulary and binary labels

2. **`extract_features.py`** - Extracts language model activations from OpenVLA
   - Loads OpenVLA model and processes instructions
   - Extracts hidden states from language encoder layers
   - Supports multi-layer feature extraction

3. **`train_probes.py`** - Trains linear probes to predict concepts
   - Trains logistic regression classifiers per concept
   - Evaluates with accuracy, F1, and average precision
   - Generates detailed results by concept category

## Quick Start

### Prerequisites

```bash
# Install dependencies
cd /umbc/rs/pi_gokhale/users/shivank2/shivanand/openvla
pip install -r concept_analysis/requirements.txt
python -m spacy download en_core_web_sm
```

## Option 1: Run Everything (End-to-End)

```bash
# Set your data path
export DATA_ROOT=/path/to/bridge_orig

# Run the full pipeline
bash concept_analysis/run_rq1_language.sh
```

This will:
1. Extract concepts from instructions → `results/concept_vocabulary.txt`
2. Extract language features from OpenVLA → `results/language_features_train.npz`
3. Train linear probes → `results/probes/probe_results_layer_-1.json`

### Option 2: Run Step-by-Step

#### Step 1: Extract Concepts

```bash
python -m concept_analysis.extract_concepts \
  --data_root /path/to/bridge_orig \
  --output_dir ./concept_analysis/results \
  --min_frequency 10 \
  --split train
```

**Output:**
- `concept_vocabulary.txt` - List of all concepts
- `concept_counts.json` - Frequency of each concept
- `concept_labels_train.npz` - Binary labels [num_instructions x num_concepts]

#### Step 2: Extract Language Features

```bash
python -m concept_analysis.extract_features \
  --model_name openvla/openvla-7b \
  --data_root /path/to/bridge_orig \
  --output_dir ./concept_analysis/results \
  --split train \
  --batch_size 8 \
  --max_samples 10000 \
  --layers "-1"
```

**Output:**
- `language_features_train.npz` - Features [num_instructions x hidden_dim]

**Note:** Use `--max_samples 10000` for quick testing, remove for full dataset

#### Step 3: Train Probes

```bash
python -m concept_analysis.train_probes \
  --features_path ./concept_analysis/results/language_features_train.npz \
  --labels_path ./concept_analysis/results/concept_labels_train.npz \
  --output_dir ./concept_analysis/results/probes \
  --layer "layer_-1" \
  --regularization 1.0 \
  --val_split 0.2
```

**Output:**
- `probe_results_layer_-1.json` - Detailed results per concept
- `probe_weights_layer_-1.npz` - Trained probe weights

## Expected Results

### Concept Vocabulary

You should get ~100-300 concepts (depending on `min_frequency`):
- **Objects**: cup, bowl, plate, block, spoon, etc.
- **Actions**: pick, place, move, grasp, push, etc.
- **Spatial**: left, right, above, below, on, etc.
- **Attributes**: red, blue, large, small, etc.

### Probe Accuracy

Expected performance:
- **Average accuracy**: 75-85%
- **Top concepts**: 95%+ (common words like "pick", "cup")
- **Rare concepts**: 60-70%

### Results Format

```json
{
  "concept": "action:pick",
  "train_pos": 5234,
  "val_pos": 1283,
  "accuracy": 0.923,
  "f1": 0.891,
  "average_precision": 0.945
}
```

## Advanced Usage

### Extract Multiple Layers

```bash
# Extract from last 3 layers
python -m concept_analysis.extract_features \
  --layers "-3,-2,-1" \
  ...
```

Then train probes for each layer:

```bash
for layer in layer_-3 layer_-2 layer_-1; do
  python -m concept_analysis.train_probes \
    --layer $layer \
    ...
done
```

### Custom Concept Filtering

Edit `extract_concepts.py` to modify:
- `min_frequency` - Minimum concept occurrences
- `spatial_relations` - Add/remove spatial concepts
- `action_primitives` - Add/remove action concepts

### Batch Processing

Process full dataset in chunks:

```bash
for i in {0..60000..10000}; do
  python -m concept_analysis.extract_features \
    --max_samples 10000 \
    --skip_samples $i \
    ...
done
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'spacy'`
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### `CUDA out of memory`
- Reduce `--batch_size` (try 4 or 2)
- Use CPU: Set `CUDA_VISIBLE_DEVICES=""` before running

### `Dataset not found`
- Verify Bridge V2 is renamed to `bridge_orig/`
- Check `--data_root` path is correct

## Next Steps

1. **Analyze Results**: Review top/bottom performing concepts
2. **Layer Comparison**: Train on multiple layers, compare concept emergence
3. **Test Set Evaluation**: Run on test split for final metrics
4. **Visualization**: Create confusion matrices, concept correlation plots
5. **TCAV Analysis**: Use probes for concept activation vectors (RQ1 Part 2)

## File Structure

```
concept_analysis/
├── __init__.py
├── README.md
├── requirements.txt
├── extract_concepts.py       # Step 1: Concept extraction
├── extract_features.py        # Step 2: Feature extraction
├── train_probes.py            # Step 3: Probe training
├── run_rq1_language.sh        # End-to-end script
└── results/                   # Output directory
    ├── concept_vocabulary.txt
    ├── concept_labels_train.npz
    ├── language_features_train.npz
    └── probes/
        ├── probe_results_layer_-1.json
        └── probe_weights_layer_-1.npz
```

## Contact / Issues

If you encounter issues:
1. Check data paths are correct
2. Verify dependencies are installed
3. Try with `--max_samples 100` first for quick debugging
4. Check GPU memory usage with `nvidia-smi`
