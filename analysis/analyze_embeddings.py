#!/usr/bin/env python3
"""
Analyze speaker embeddings (x-vectors) from CapSpeech
Handles Kaldi-format embeddings (.ark and .csv files)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import struct

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
EMBEDDING_PATH = PROJECT_ROOT / "embeddings"
DATA_PATH = PROJECT_ROOT / "data"

print("="*80)
print("Speaker Embedding (X-Vector) Analysis")
print("="*80)

# Check if embeddings are mounted
print("\n[1] Checking embedding directory...")
if not EMBEDDING_PATH.exists():
    print(f"✗ Embedding path not found: {EMBEDDING_PATH}")
    print("\nPlease mount embeddings first:")
    print("sshfs psingh54@login.clsp.jhu.edu:/home/tthebau1/SHADOW/.../CapSpeech-real ./embeddings")
    sys.exit(1)

# List files
files = list(EMBEDDING_PATH.glob("*"))
print(f"✓ Embedding directory exists: {EMBEDDING_PATH}")
print(f"✓ Found {len(files)} files")

csv_files = list(EMBEDDING_PATH.glob("*.csv"))
ark_files = list(EMBEDDING_PATH.glob("*.ark"))

print(f"\n  CSV files: {len(csv_files)}")
for f in csv_files[:5]:
    print(f"    • {f.name}")

print(f"\n  ARK files: {len(ark_files)}")
for f in ark_files[:5]:
    print(f"    • {f.name}")

# Analyze CSV files (metadata)
print("\n[2] Analyzing CSV metadata files...")
all_xvector_data = []

for csv_file in csv_files:
    try:
        print(f"\n  Reading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        print(f"    ✓ Loaded {len(df)} entries")
        print(f"    ✓ Columns: {list(df.columns)}")

        if len(df) > 0:
            print(f"    ✓ Sample rows:")
            print(df.head(3).to_string(index=False))
            all_xvector_data.append(df)
    except Exception as e:
        print(f"    ✗ Error reading {csv_file.name}: {e}")

if all_xvector_data:
    combined_df = pd.concat(all_xvector_data, ignore_index=True)
    print(f"\n✓ Total x-vector entries across all CSV files: {len(combined_df)}")

# Function to read Kaldi ARK file (binary format)
def read_kaldi_ark(ark_file, max_vectors=10):
    """
    Read Kaldi ARK file and extract x-vectors
    Returns dict: {utterance_id: vector}
    """
    vectors = {}

    try:
        with open(ark_file, 'rb') as f:
            count = 0
            while count < max_vectors:  # Limit to first N vectors for analysis
                # Read utterance ID (text until space)
                utt_id = b''
                while True:
                    char = f.read(1)
                    if not char or char == b' ':
                        break
                    utt_id += char

                if not utt_id:
                    break

                utt_id = utt_id.decode('utf-8')

                # Skip binary header marker
                f.read(2)  # '\0B'

                # Read dimension
                f.read(1)  # '\4'
                dim = struct.unpack('i', f.read(4))[0]

                # Read vector
                vector = np.frombuffer(f.read(dim * 4), dtype=np.float32)

                vectors[utt_id] = vector
                count += 1

                if count == 1:
                    print(f"      First vector ID: {utt_id}, Dimension: {dim}")

        return vectors
    except Exception as e:
        print(f"      Note: Binary ARK parsing encountered: {e}")
        print(f"      (This is expected - ARK files are complex Kaldi binary format)")
        return {}

# Try to analyze ARK files (sample only)
print("\n[3] Analyzing ARK files (sample)...")
print("  Note: ARK files are Kaldi binary format. Showing structural analysis only.")

for ark_file in ark_files[:2]:  # Just check first 2
    print(f"\n  {ark_file.name}:")
    file_size = ark_file.stat().st_size
    print(f"    • File size: {file_size / (1024**2):.2f} MB")

    # Try to read a few vectors
    print(f"    • Attempting to read sample vectors...")
    sample_vectors = read_kaldi_ark(ark_file, max_vectors=3)

    if sample_vectors:
        print(f"    ✓ Successfully parsed {len(sample_vectors)} sample vectors")
        dims = [len(v) for v in sample_vectors.values()]
        print(f"    ✓ Vector dimensions: {dims}")

        # Show statistics of first vector
        first_vec = list(sample_vectors.values())[0]
        print(f"    ✓ Sample vector stats:")
        print(f"      - Mean: {first_vec.mean():.4f}")
        print(f"      - Std:  {first_vec.std():.4f}")
        print(f"      - Min:  {first_vec.min():.4f}")
        print(f"      - Max:  {first_vec.max():.4f}")

# Check alignment with dataset
print("\n[4] Checking alignment with CapSpeech dataset...")
try:
    from datasets import load_from_disk
    train_ds = load_from_disk(f"{DATA_PATH}/train")

    print(f"✓ Loaded dataset: {len(train_ds)} samples")

    # Check if audio_path matches embedding IDs
    print("\n  Sample audio paths from dataset:")
    for i in range(5):
        audio_path = train_ds[i]['audio_path']
        print(f"    • {audio_path}")

    # Extract base filename (likely the embedding key)
    sample_audio_path = train_ds[0]['audio_path']
    base_name = Path(sample_audio_path).stem
    print(f"\n  Example base filename (likely embedding key): {base_name}")

    if all_xvector_data:
        # Check if this ID exists in CSV
        if 'utt_id' in combined_df.columns or 'utterance_id' in combined_df.columns:
            id_col = 'utt_id' if 'utt_id' in combined_df.columns else 'utterance_id'
            if base_name in combined_df[id_col].values:
                print(f"  ✓ Found matching ID in CSV!")
            else:
                print(f"  ⚠ ID not found in CSV. Checking first few CSV IDs:")
                print(f"    {list(combined_df[id_col].head(5))}")

except Exception as e:
    print(f"✗ Could not load dataset: {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if csv_files:
    print(f"✓ CSV metadata files: {len(csv_files)} files")
    if all_xvector_data:
        print(f"✓ Total x-vector entries: {len(combined_df)}")

if ark_files:
    print(f"✓ ARK binary files: {len(ark_files)} files")
    total_ark_size = sum(f.stat().st_size for f in ark_files) / (1024**3)
    print(f"✓ Total ARK size: {total_ark_size:.2f} GB")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
To work with these embeddings:

1. RECOMMENDED: Use Kaldi toolkit or kaldiio library
   pip install kaldiio

   Example code:
   import kaldiio
   xvectors = kaldiio.load_ark('xvector.ark')

2. ALTERNATIVE: Extract embeddings to numpy format
   - Use Kaldi's copy-vector command
   - Or write custom parser

3. CREATE MAPPING: Build audio_path -> embedding lookup
   - Match dataset audio_path to embedding utterance IDs
   - Save as pickle/json for fast loading during training

4. VERIFY: Check embedding dimensions and statistics
   - Ensure all embeddings have same dimension
   - Normalize if needed for your model
""")

print("="*80)
