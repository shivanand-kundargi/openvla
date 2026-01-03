
import os
import json
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def convert_to_images(
    data_root: str = None, 
    max_samples: int = 1000, 
    output_dir: str = "./concept_analysis/data/images"
):
    """
    Reads Bridge V2 from TFDS and converts it to a folder of images + metadata.json.
    This runs ONCE.
    """
    # 1. Setup Output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f" Converting Bridge V2 -> Images in {output_path}")
    
    # 2. Load TFDS (The messy part)
    # We use the big disk cache if set via env var
    split = "train"
    ds = None
    
    try:
        # User specified path or default project path
        # The data is in concept_analysis/data/bridge_dataset/1.0.0
        # So data_dir must be concept_analysis/data
        search_paths = [
            data_root,
            "./concept_analysis/data",
            "/umbc/rs/pi_gokhale/users/shivank2/shivanand/openvla/concept_analysis/data"
        ]
        
        for p in search_paths:
            if p and os.path.exists(p):
                print(f"Checking {p}...")
                try:
                    ds = tfds.load("bridge_dataset", data_dir=p, split=split)
                    print(f"✓ Found data in {p}")
                    break
                except Exception as e:
                    print(f"  Not found in {p}: {e}")
                    
    except Exception:
        pass
    
    if ds is None:
        raise ValueError("Could not load Bridge V2 to convert it!")

    # 3. Iterate and Save
    metadata = {}
    
    count = 0
    for i, episode in enumerate(tqdm(ds, desc="Converting")):
        if count >= max_samples:
            break
            
        try:
            # Safe iterator access
            steps = iter(episode["steps"])
            
            # Step 1
            step = next(steps)
            
            # a. Instruction
            instruction = step["language_instruction"].numpy().decode("utf-8")
            
            # b. Image
            img_tensor = step["observation"]["image_0"]
            img_array = img_tensor.numpy()
            image = Image.fromarray(img_array)
            
            # c. Action (Avg 5 steps)
            step_actions = [step["action"].numpy()]
            for _ in range(4):
                try:
                    next_step = next(steps)
                    step_actions.append(next_step["action"].numpy())
                except StopIteration:
                    break
            avg_action = np.mean(step_actions, axis=0).tolist() # Convert to list for JSON
            
            # Save Image
            filename = f"image_{count:05d}.jpg"
            image.save(output_path / filename)
            
            # Save Metadata
            metadata[filename] = {
                "instruction": instruction,
                "action": avg_action,
                "index": count
            }
            
            count += 1
            
        except Exception as e:
            continue
            
    # 4. Save JSON
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"✓ Converted {count} samples to {output_path}")

if __name__ == "__main__":
    # Ensure TFDS uses big disk
    project_tfds = os.path.abspath("./concept_analysis/data/tfds")
    os.environ["TFDS_DATA_DIR"] = project_tfds
    
    convert_to_images(
        data_root=project_tfds,
        max_samples=2000 # Convert enough for robust testing
    )
