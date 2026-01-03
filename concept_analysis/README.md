# OpenVLA Linguistic Concept Analysis

**Project:** Analyzing the internal semantic grounding of Vision-Language-Action models.

This repository contains the complete pipeline for analyzing how OpenVLA represents linguistic concepts (objects, actions, attributes) and how those internal representations correlate with physical robot control.

---

## 🚀 Key Features

*   **Automated Content Extraction:** Parses robot instructions into 3,000+ atomic concepts (e.g., `obj:apple`, `action:pick`) using NLP.
*   **Multilayer Probing:** Extracts activations from OpenVLA Layers 0-32 and trains linear probes to test semantic decodability.
*   **Alignment Analysis:** Correlates internal concept activation with ground-truth robot actions to measure "Alignment Gap".
*   **Zero-Shot Evaluation:** Tests the model on the Bridge V2 dataset without fine-tuning.

---

## 📂 Project Structure

```
concept_analysis/
├── data/               # Dataset handling and utilities
├── results/            # Artifacts (extracted concepts, feature arrays, plots)
│   └── multilayer/     # Results for layer-wise analysis
├── extract_concepts.py # Step 1: Parse instructions -> Concept Labels
├── extract_features.py # Step 2: Run Model -> Save Hidden States
├── train_probes.py     # Step 3: Train Classifiers -> Probe Accuracy
├── analyze_results.py  # Step 4: Generate Plots & Correlation Analysis
└── research_report.md  # Detailed write-up of findings
```

---

## 🛠️ Installation

**Requirements:**
*   OpenVLA environment (PyTorch, Transformers, etc.)
*   `spaCy` for linguistic parsing

```bash
# Install spaCy dependencies
pip install spacy
python -m spacy download en_core_web_sm

# Install analysis libs
pip install scikit-learn matplotlib seaborn pandas tqdm
```

---

## ⚡ Quick Start Pipeline

Run the full analysis in 4 sequential steps:

### 1. Extract Concepts
Parse the instructions from the Bridge V2 dataset to create binary concept labels.

```bash
python -m concept_analysis.extract_concepts \
  --data_root /path/to/bridge_orig \
  --output_dir ./concept_analysis/results
```
*Outputs: `concept_vocabulary.txt`, `concept_labels_train.npz`*

### 2. Extract Features
Run OpenVLA on the dataset (images + text) and save the hidden states for specific layers.

```bash
python -m concept_analysis.extract_features \
  --model_name openvla/openvla-7b \
  --data_root /path/to/bridge_orig \
  --output_dir ./concept_analysis/results \
  --layers 0 8 16 24 32
```
*Outputs: `language_features_train.npz` (or per-layer files)*

### 3. Train Probes
Train Logistic Regression probes to predict concepts from hidden states.

```bash
# Run for a specific layer (e.g., Layer 32)
python -m concept_analysis.train_probes \
  --features_path ./concept_analysis/results/multilayer/features_layer_32.npz \
  --labels_path ./concept_analysis/results/concept_labels_train.npz \
  --output_dir ./concept_analysis/results/multilayer/probes \
  --layer_idx 32
```

### 4. Analyze & Visualize
Generate accuracy plots and correlation heatmaps.

```bash
python -m concept_analysis.analyze_results \
  --results_dir ./concept_analysis/results/multilayer/probes \
  --output_dir ./concept_analysis/results/multilayer/analysis
```
*Outputs: Accuracy Plots, Correlation Heatmaps, CSV Tables*

---

## 📊 Key Findings

*   **High Semantic Understanding:** OpenVLA retains near-perfect linear information about objects (`obj:apple`) and attributes (`attr:red`) in its final layers (>99% probe accuracy).
*   **Alignment Level:** While the model *understands* the concepts, its zero-shot action predictions show **calibration errors** (e.g., inverting up/down movements) when applied to the Bridge V2 embodiment.

