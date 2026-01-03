
import tensorflow_datasets as tfds
import os

data_dir = "./concept_analysis/data"
dataset_path = os.path.join(data_dir, "bridge_dataset", "1.0.0")

print(f"Testing builder_from_directory with path: {dataset_path}")

try:
    builder = tfds.builder_from_directory(dataset_path)
    print("Success! Builder loaded.")
    print(f"Dataset info: {builder.info.name}")
except Exception as e:
    print(f"Error: {e}")
