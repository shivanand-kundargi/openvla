"""
extract_concepts.py

Extract linguistic concepts from Bridge V2 task instructions.
Creates concept vocabulary and binary labels for each instruction.

Usage:
    python -m concept_analysis.extract_concepts \\
        --data_root /path/to/bridge_orig \\
        --output_dir ./results \\
        --min_frequency 10
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

try:
    import spacy
except ImportError:
    print("Please install spacy: pip install spacy")
    print("Then download model: python -m spacy download en_core_web_sm")
    raise


class ConceptExtractor:
    """Extract linguistic concepts from robot task instructions."""
    
    def __init__(self, min_frequency: int = 10):
        self.min_frequency = min_frequency
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Robot-specific concepts
        self.spatial_relations = {
            "above", "below", "on", "under", "next", "beside", "near",
            "left", "right", "front", "back", "inside", "outside",
            "in", "into", "onto", "off", "over", "around", "through", "top", "bottom"
        }
        
        self.action_primitives = {
            "pick", "place", "put", "move", "push", "pull", "grasp", "release",
            "lift", "lower", "open", "close", "rotate", "turn", "slide",
            "stack", "insert", "remove", "flip", "pour", "take"
        }
        
        self.vocabulary = None
        self.concept_to_idx = None
        
    def extract_from_text(self, text: str) -> Set[str]:
        """Extract concepts from a single instruction."""
        concepts = set()
        text_lower = text.lower().strip()
        
        # Parse with spaCy
        doc = self.nlp(text_lower)
        
        # Extract nouns (objects)
        for token in doc:
            if token.pos_ == "NOUN" and len(token.lemma_) > 2:
                concepts.add(f"obj:{token.lemma_}")
        
        # Extract verbs (actions)
        for token in doc:
            if token.pos_ == "VERB" and len(token.lemma_) > 2:
                concepts.add(f"verb:{token.lemma_}")
        
        # Extract adjectives (attributes like colors)
        for token in doc:
            if token.pos_ == "ADJ":
                concepts.add(f"attr:{token.lemma_}")
        
        # Check for spatial relations
        words_in_text = set(text_lower.split())
        for spatial in self.spatial_relations:
            if spatial in words_in_text:
                concepts.add(f"spatial:{spatial}")
        
        # Check for action primitives
        for action in self.action_primitives:
            if action in text_lower:
                concepts.add(f"action:{action}")
        
        # Multi-word actions
        words = text_lower.split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            if bigram in {"pick_up", "put_down", "set_down", "place_on"}:
                concepts.add(f"action:{bigram}")
        
        return concepts
    
    def build_vocabulary(self, instructions: List[str]) -> Dict[str, int]:
        """Build concept vocabulary from instructions."""
        print(f"Extracting concepts from {len(instructions)} instructions...")
        
        # Count all concepts
        concept_counts = {}
        for instruction in tqdm(instructions):
            concepts = self.extract_from_text(instruction)
            for concept in concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Filter by minimum frequency
        filtered = {
            c: count for c, count in concept_counts.items()
            if count >= self.min_frequency
        }
        
        print(f"\nTotal unique concepts: {len(concept_counts)}")
        print(f"Concepts with freq >= {self.min_frequency}: {len(filtered)}")
        
        # Create vocabulary
        self.vocabulary = sorted(filtered.keys())
        self.concept_to_idx = {c: i for i, c in enumerate(self.vocabulary)}
        
        # Print stats by category
        self._print_stats(filtered)
        
        return filtered
    
    def _print_stats(self, concept_counts: Dict[str, int]):
        """Print statistics by concept category."""
        categories = {}
        for concept in concept_counts:
            category = concept.split(":")[0] if ":" in concept else "other"
            if category not in categories:
                categories[category] = []
            categories[category].append(concept)
        
        print("\nConcept vocabulary by category:")
        print("-" * 60)
        for cat in sorted(categories.keys()):
            concepts = categories[cat]
            print(f"  {cat:10s}: {len(concepts):4d} concepts")
            # Show top 5 examples
            examples = sorted(concepts, key=lambda c: concept_counts[c], reverse=True)[:5]
            for ex in examples:
                print(f"      {ex:30s} ({concept_counts[ex]:4d} occurrences)")
        print("-" * 60)
    
    def create_labels(self, instructions: List[str]) -> np.ndarray:
        """Create binary label matrix [num_instructions x num_concepts]."""
        if self.vocabulary is None:
            raise ValueError("Must call build_vocabulary() first!")
        
        num_inst = len(instructions)
        num_concepts = len(self.vocabulary)
        labels = np.zeros((num_inst, num_concepts), dtype=np.float32)
        
        print(f"\nCreating label matrix: {num_inst} x {num_concepts}")
        for i, instruction in enumerate(tqdm(instructions)):
            concepts = self.extract_from_text(instruction)
            for concept in concepts:
                if concept in self.concept_to_idx:
                    j = self.concept_to_idx[concept]
                    labels[i, j] = 1.0
        
        # Stats
        avg_per_inst = labels.sum(axis=1).mean()
        avg_per_concept = labels.sum(axis=0).mean()
        sparsity = 1.0 - (labels.sum() / labels.size)
        
        print(f"\nLabel matrix statistics:")
        print(f"  Avg concepts per instruction: {avg_per_inst:.2f}")
        print(f"  Avg instructions per concept: {avg_per_concept:.1f}")
        print(f"  Sparsity: {sparsity:.1%}")
        
        return labels


def find_dataset_path(root_dir: str) -> str:
    """Recursively find directory containing dataset_info.json."""
    if not os.path.exists(root_dir):
        return None
    
    # Check current directory
    if os.path.exists(os.path.join(root_dir, "dataset_info.json")):
        return root_dir
        
    # Check children
    for root, dirs, files in os.walk(root_dir):
        if "dataset_info.json" in files:
            return root
    return None


import glob

def load_from_tfrecords_direct(data_root: str, max_samples: int = None) -> List[str]:
    """Manually parse TFRecords to extract instructions (Bypass TFDS)."""
    print(f"Attempting direct TFRecord parsing from {data_root}...")
    
    # Find tfrecord files
    files = glob.glob(os.path.join(data_root, "**", "*.tfrecord*"), recursive=True)
    if not files:
        print("No .tfrecord files found recursively.")
        return None
        
    print(f"Found {len(files)} tfrecord files. reading...")
    
    instructions = []
    
    # Create dataset
    ds = tf.data.TFRecordDataset(files)
    
    count = 0
    for raw_record in tqdm(ds, desc="Parsing TFRecords"):
        if max_samples and count >= max_samples:
            break
            
        try:
            ex = tf.train.SequenceExample()
            ex.ParseFromString(raw_record.numpy())
            
            # Find language key
            fl = ex.feature_lists.feature_list
            lang_key = next((k for k in fl.keys() if 'language' in k), None)
            
            if lang_key:
                # Get first step's instruction
                # feature[0] is the first step
                bs = fl[lang_key].feature[0].bytes_list.value[0]
                instructions.append(bs.decode('utf-8'))
                count += 1
        except Exception as e:
            pass # Skip bad records
            
    print(f"Directly parsed {len(instructions)} instructions")
    return instructions


def load_bridge_instructions(data_root: str = None, split: str = "train") -> List[str]:
    """Load task instructions from Bridge V2 dataset."""
    
    # Priority 1: Load from converted Image Metadata (Pure JSON, no TFDS)
    # This aligns with the new "Image Pipeline"
    metadata_path = Path("./concept_analysis/data/images/metadata.json")
    if metadata_path.exists():
        print(f"Loading instructions from metadata: {metadata_path}")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Sort by filename index to align with other scripts
            sorted_keys = sorted(metadata.keys())
            instructions = [metadata[k]["instruction"] for k in sorted_keys]
            
            print(f"✓ Loaded {len(instructions)} instructions from metadata")
            return instructions
        except Exception as e:
            print(f"Failed to read metadata: {e}")

    # Priority 2: Legacy TFDS Load (Fallback)
    print("Metadata not found. Falling back to TFDS...")
    ds = None
    
    if data_root:
        data_root = os.path.abspath(data_root)
        print(f"Searching for dataset in {data_root}...")
        
        # Try 1: builder_from_directory
        dataset_path = find_dataset_path(data_root)
        if dataset_path:
            try:
                builder = tfds.builder_from_directory(dataset_path)
                ds = builder.as_dataset(split=split)
            except Exception as e:
                print(f"builder_from_directory failed: {e}")

        # Try 2: tfds.load
        if ds is None:
             try:
                ds = tfds.load("bridge_dataset", data_dir=data_root, split=split)
             except Exception:
                pass
    
    # Try 3: Direct TFRecord parsing (Fallback)
    if ds is None and data_root:
        print("TFDS loading failed. Trying direct TFRecord loading...")
        instructions = load_from_tfrecords_direct(data_root)
        if instructions:
            return instructions
            
    if ds is None and not data_root:
         print(f"Loading Bridge V2 {split} split (auto-download from tfds)...")
         ds = tfds.load("bridge_dataset", split=split, download=True)
            
    if ds is None:
        raise ValueError("Could not load dataset via any method. Check paths.")
    
    # If we got a TFDS dataset, iterate it
    instructions = []
    for episode in tqdm(ds):
        steps = episode["steps"]
        first_step = next(iter(steps))
        instruction = first_step["language_instruction"].numpy().decode("utf-8")
        instructions.append(instruction)
    
    print(f"Loaded {len(instructions)} instructions")
    return instructions
    
    # Extract instructions
    instructions = []
    for episode in tqdm(ds):
        # Get instruction from first step (steps is a Dataset, not a list)
        steps = episode["steps"]
        first_step = next(iter(steps))
        instruction = first_step["language_instruction"].numpy().decode("utf-8")
        instructions.append(instruction)
    
    print(f"Loaded {len(instructions)} instructions")
    return instructions


def main():
    parser = argparse.ArgumentParser(description="Extract concepts from Bridge V2 instructions")
    parser.add_argument("--data_root", type=str, default=None,
                       help="Path to bridge_orig dataset directory (optional, will auto-download if not provided)")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for saved files")
    parser.add_argument("--min_frequency", type=int, default=10,
                       help="Minimum concept frequency to include in vocabulary")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to use (train/test)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load instructions
    instructions = load_bridge_instructions(args.data_root, args.split)
    
    # Extract concepts
    extractor = ConceptExtractor(min_frequency=args.min_frequency)
    concept_counts = extractor.build_vocabulary(instructions)
    
    # Create labels
    labels = extractor.create_labels(instructions)
    
    # Save outputs
    print(f"\nSaving outputs to {output_dir}/...")
    
    # Save vocabulary
    vocab_file = output_dir / "concept_vocabulary.txt"
    with open(vocab_file, "w") as f:
        for concept in extractor.vocabulary:
            f.write(f"{concept}\n")
    print(f"  Saved vocabulary: {vocab_file}")
    
    # Save concept counts
    counts_file = output_dir / "concept_counts.json"
    with open(counts_file, "w") as f:
        json.dump(concept_counts, f, indent=2)
    print(f"  Saved counts: {counts_file}")
    
    # Save labels
    labels_file = output_dir / f"concept_labels_{args.split}.npz"
    np.savez_compressed(
        labels_file,
        labels=labels,
        instructions=np.array(instructions, dtype=object),
        vocabulary=np.array(extractor.vocabulary, dtype=object)
    )
    print(f"  Saved labels: {labels_file}")
    
    # Save extractor
    extractor_file = output_dir / "concept_extractor.pkl"
    with open(extractor_file, "wb") as f:
        pickle.dump(extractor, f)
    print(f"  Saved extractor: {extractor_file}")
    
    print("\n✓ Done!")
    print(f"\nNext step: Extract language features from OpenVLA")
    print(f"  python -m concept_analysis.extract_features --help")


if __name__ == "__main__":
    main()
