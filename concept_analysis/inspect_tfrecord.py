
import tensorflow as tf
import glob
import os

# Find tfrecords
# Try recursively finding if path is not exact
files = glob.glob("concept_analysis/data/bridge_dataset/1.0.0/*.tfrecord*")
if not files:
    print("No tfrecords found in default path, searching recursively...")
    files = glob.glob("concept_analysis/data/**/*.tfrecord*", recursive=True)

if not files:
    print("ERROR: No .tfrecord files found!")
    exit(1)

print(f"Found {len(files)} tfrecord files.")
print(f"Reading first file: {files[0]}")

ds = tf.data.TFRecordDataset(files[0])

print("\n--- First Example Structure ---")
for proto in ds.take(1):
    try:
        # Try SequenceExample (most RLDS)
        ex = tf.train.SequenceExample()
        ex.ParseFromString(proto.numpy())
        print("Successfully parsed as SequenceExample")
        
        print("Context Features:")
        print(ex.context.feature.keys())
        
        print("\nFeature Lists (Steps):")
        print(ex.feature_lists.feature_list.keys())
        
        # Check language
        if 'language_instruction' in ex.feature_lists.feature_list: # might be 'steps/language_instruction'
             print("\nFound 'language_instruction'!")
        
        # RLDS often flattens keys like 'steps/observation/image'
        # Let's print all keys
        
    except Exception as e:
        print(f"Error parsing as SequenceExample: {e}")
        # Try Example
        try:
            ex = tf.train.Example()
            ex.ParseFromString(proto.numpy())
            print("Successfully parsed as Example")
            print(ex.features.feature.keys())
        except Exception as e2:
             print(f"Error parsing as Example: {e2}")

print("\n-------------------------------")
