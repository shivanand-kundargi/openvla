"""
extract_features.py

Extract language model hidden states from OpenVLA for each instruction.
These features will be used to train linear concept probes.
"""

import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from pathlib import Path

# NOTE: NO TENSORFLOW IMPORTS HERE! Pure PyTorch.

class LanguageFeatureExtractor:
    """Extract language encoder features from OpenVLA."""
    def __init__(self, model_name="openvla/openvla-7b", device="cuda"):
        self.device = device
        print(f"Loading OpenVLA model: {model_name}")
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)
        self.model.eval()
        print("✓ Model loaded")

    @torch.no_grad()
    def extract_batch(self, batch_data, layers_to_extract=[-1]):
        """
        Extract features for a batch of (instruction, image) pairs.
        """
        instructions = [b["instruction"] for b in batch_data]
        images = [b["image"] for b in batch_data]
        
        # OpenVLA Prompt Format
        prompts = [f"In: What action should the robot take to {inst.lower()}?\nOut:" for inst in instructions]
        
        # Prepare inputs
        inputs = self.processor(prompts, images, padding=True).to(self.device, dtype=torch.bfloat16)
        
        # Forward pass
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=False
        )
        
        hidden_states = outputs.hidden_states
        
        batch_results = {f"layer_{l}": [] for l in layers_to_extract}
        
        for l in layers_to_extract:
             # [B, Seq, Dim] -> [B, Dim] (Mean Pooling)
             # Note: This pools over all tokens (vision + language). 
             # For stricter language analysis, we might want to mask, but mean pooling is standard baseline.
             layer_feat = hidden_states[l].mean(dim=1).float().cpu().numpy()
             for i in range(len(batch_data)):
                 batch_results[f"layer_{l}"].append(layer_feat[i])
                 
        # Action Prediction (Greedy Generation)
        # OpenVLA only supports batch_size=1 for generation currently
        predictions = []
        input_len = inputs.input_ids.shape[1]
        
        for i in range(len(batch_data)):
            # Slice single item
            single_input = {k: v[i:i+1] for k, v in inputs.items()}
            
            generated_ids = self.model.generate(
                **single_input, 
                max_new_tokens=7, 
                do_sample=False
            )
            # Save raw tokens (Pad to length 7)
            raw_tokens = generated_ids[0, input_len:].cpu().numpy()
            
            # Pad with zeros if shorter than 7
            padded_action = np.zeros((7,), dtype=raw_tokens.dtype)
            length = min(len(raw_tokens), 7)
            padded_action[:length] = raw_tokens[:length]
            
            predictions.append(padded_action)
                 
        return batch_results, predictions

def load_image_dataset(image_dir, max_samples=None):
    """Load metadata and images from directory."""
    image_path = Path(image_dir)
    json_path = image_path / "metadata.json"
    
    if not json_path.exists():
        # Fallback check
        alt_path = Path("concept_analysis/data/images/metadata.json")
        if alt_path.exists():
             json_path = alt_path
             image_path = alt_path.parent
        else:
            raise FileNotFoundError(f"Metadata not found at {json_path}. Did you run convert_bridge_to_images.py?")
        
    with open(json_path, "r") as f:
        metadata = json.load(f)
        
    dataset = []
    print(f"Loading images from {image_path}...")
    
    # Sort keys to ensure deterministic order
    sorted_keys = sorted(metadata.keys())
    if max_samples:
        sorted_keys = sorted_keys[:max_samples]
        
    for filename in tqdm(sorted_keys):
        info = metadata[filename]
        try:
            img = Image.open(image_path / filename).convert("RGB")
            dataset.append({
                "instruction": info["instruction"],
                "image": img,
                "action": np.array(info["action"]), # Ground Truth
                "filename": filename
            })
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            
    print(f"✓ Loaded {len(dataset)} samples.")
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openvla/openvla-7b")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--layers", type=str, default="-1")
    # Ignored args to keep script compatible
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    
    args = parser.parse_args()
    
    # 1. Load Clean Image Data
    image_dir = "./concept_analysis/data/images"
    dataset = load_image_dataset(image_dir, args.max_samples)
    
    if not dataset:
        raise ValueError("No data loaded!")
    
    # 2. Extract Features
    layers = [int(x) for x in args.layers.split(",")]
    extractor = LanguageFeatureExtractor(args.model_name)
    
    all_features = {f"layer_{l}": [] for l in layers}
    all_instructions = []
    all_actions = []
    all_predictions = []
    
    # Batch Processing
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Extracting"):
        batch = dataset[i : i + args.batch_size]
        
        feats, preds = extractor.extract_batch(batch, layers)
        
        for k, v in feats.items():
            all_features[k].extend(v)
            
        all_instructions.extend([b["instruction"] for b in batch])
        all_actions.extend([b["action"] for b in batch])
        all_predictions.extend(preds)
        
    # 3. Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "language_features_train.npz"
    
    save_dict = {k: np.array(v) for k, v in all_features.items()}
    save_dict["instructions"] = np.array(all_instructions, dtype=object)
    save_dict["actions"] = np.array(all_actions)
    save_dict["predictions"] = np.array(all_predictions) # [N, 7] tokens
    save_dict["layers"] = np.array(layers)
    
    print(f"Saving to {output_file}")
    np.savez_compressed(output_file, **save_dict)
    print("Done!")

if __name__ == "__main__":
    main()
