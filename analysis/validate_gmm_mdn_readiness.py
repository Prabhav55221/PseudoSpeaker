#!/usr/bin/env python3
"""
Pre-flight validation for GMM-MDN training
Ensures all data is accessible and properly aligned
"""

import sys
import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from datasets import load_from_disk

print("="*80)
print("GMM-MDN Training Readiness Validation")
print("="*80)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
EMBEDDING_PATH = PROJECT_ROOT / "embeddings"

# ============================================================================
# [1] Load Dataset
# ============================================================================
print("\n[1] Loading CapSpeech dataset...")
try:
    train_ds = load_from_disk(f"{DATA_PATH}/train")
    val_ds = load_from_disk(f"{DATA_PATH}/val")
    test_ds = load_from_disk(f"{DATA_PATH}/test")
    print(f"✓ Train: {len(train_ds):,} samples")
    print(f"✓ Val:   {len(val_ds):,} samples")
    print(f"✓ Test:  {len(test_ds):,} samples")
except Exception as e:
    print(f"✗ Failed to load dataset: {e}")
    sys.exit(1)

# ============================================================================
# [2] Load Embedding Metadata
# ============================================================================
print("\n[2] Loading embedding metadata...")
try:
    csv_files = list(EMBEDDING_PATH.glob("*.csv"))
    if not csv_files:
        print(f"✗ No CSV files found in {EMBEDDING_PATH}")
        print("Please ensure embeddings are mounted!")
        sys.exit(1)

    all_xvector_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_xvector_dfs.append(df)

    xvector_metadata = pd.concat(all_xvector_dfs, ignore_index=True)
    print(f"✓ Loaded {len(xvector_metadata):,} x-vector metadata entries")
    print(f"✓ Columns: {list(xvector_metadata.columns)}")

    # Create lookup: audio_id -> (ark_file, byte_offset)
    xvector_lookup = {}
    for _, row in xvector_metadata.iterrows():
        xvector_lookup[row['id']] = {
            'storage_path': row['storage_path'],
            'storage_byte': row['storage_byte'],
            'speech_duration': row['speech_duration']
        }

    print(f"✓ Created lookup table with {len(xvector_lookup):,} entries")

except Exception as e:
    print(f"✗ Failed to load embeddings: {e}")
    sys.exit(1)

# ============================================================================
# [3] Check Data Alignment
# ============================================================================
print("\n[3] Checking data alignment (audio_path → embeddings)...")

aligned_count = 0
missing_count = 0
sample_matches = []

for i in range(min(100, len(train_ds))):
    audio_path = train_ds[i]['audio_path']

    if audio_path in xvector_lookup:
        aligned_count += 1
        if len(sample_matches) < 5:
            sample_matches.append((audio_path, xvector_lookup[audio_path]))
    else:
        missing_count += 1
        if missing_count <= 3:
            print(f"  ⚠ Missing embedding for: {audio_path}")

alignment_rate = aligned_count / min(100, len(train_ds)) * 100
print(f"✓ Alignment rate (first 100): {alignment_rate:.1f}%")
print(f"✓ Matched: {aligned_count}, Missing: {missing_count}")

if alignment_rate < 90:
    print(f"✗ WARNING: Low alignment rate! Expected >95%")
else:
    print(f"✓ Alignment looks good!")

print("\n  Sample matches:")
for audio_path, metadata in sample_matches[:3]:
    print(f"    {audio_path}")
    print(f"      → {Path(metadata['storage_path']).name} @ byte {metadata['storage_byte']}")

# Full dataset alignment check
print("\n  Checking full training set alignment...")
full_aligned = sum(1 for ex in train_ds if ex['audio_path'] in xvector_lookup)
full_alignment_rate = full_aligned / len(train_ds) * 100
print(f"✓ Full training set alignment: {full_alignment_rate:.2f}% ({full_aligned:,}/{len(train_ds):,})")

if full_alignment_rate < 95:
    print("  ✗ WARNING: Some audio files missing embeddings!")
    print("  This might be okay if we have enough data, but investigate if rate < 90%")

# ============================================================================
# [4] Check Embedding Files (Skip Loading - Will Use Hyperion on Cluster)
# ============================================================================
print("\n[4] Checking embedding file structure...")
print("  Note: Actual embedding loading will use Hyperion on cluster")

# Just check file existence and structure
ark_files = list(EMBEDDING_PATH.glob("*.ark"))
print(f"✓ Found {len(ark_files)} ARK files")

if ark_files:
    total_size = sum(f.stat().st_size for f in ark_files)
    print(f"✓ Total ARK size: {total_size / (1024**3):.2f} GB")
    print(f"  Sample ARK files:")
    for f in ark_files[:3]:
        print(f"    - {f.name} ({f.stat().st_size / (1024**2):.1f} MB)")
else:
    print("  ⚠ No ARK files found (okay if running locally)")

print("\n  Embedding loading will be handled by:")
print("    - Hyperion library on cluster")
print("    - Link: https://hyperion-ml.readthedocs.io/en/latest/")
print("    - Skipping local embedding tests")

# ============================================================================
# [5] Filter Dataset to Sources with Embeddings
# ============================================================================
print("\n[5] Filtering dataset to sources with embeddings...")

# Sources with embeddings (from alignment analysis)
sources_with_embeddings = ['voxceleb', 'ears', 'expresso']

# Filter dataset
filtered_indices = []
for i, ex in enumerate(train_ds):
    if ex.get('source') in sources_with_embeddings:
        # Also check if embedding exists
        if ex['audio_path'] in xvector_lookup:
            filtered_indices.append(i)

print(f"✓ Original dataset: {len(train_ds):,} samples")
print(f"✓ Filtered dataset: {len(filtered_indices):,} samples")
print(f"✓ Filtered to {len(filtered_indices)/len(train_ds)*100:.1f}% of original")

# ============================================================================
# [6] Create Attribute Grouping (on filtered data)
# ============================================================================
print("\n[6] Creating attribute grouping for GMM-MDN training...")

# Function to create attribute text
def create_attribute_text(example):
    """Convert structured attributes to standardized text"""
    parts = []

    if example.get('gender'):
        parts.append(example['gender'])

    if example.get('age'):
        parts.append(example['age'])

    if example.get('pitch'):
        parts.append(example['pitch'])

    if example.get('speaking_rate'):
        parts.append(example['speaking_rate'])

    return ', '.join(parts) if parts else 'unknown'

# Group by attributes (filtered data only)
attribute_groups = defaultdict(list)
attribute_text_to_attrs = {}

for idx in filtered_indices:
    ex = train_ds[idx]

    # Create standardized attribute text
    attr_text = create_attribute_text(ex)

    # Store the mapping
    if attr_text not in attribute_text_to_attrs:
        attribute_text_to_attrs[attr_text] = {
            'gender': ex.get('gender'),
            'age': ex.get('age'),
            'pitch': ex.get('pitch'),
            'speaking_rate': ex.get('speaking_rate')
        }

    # Group samples
    attribute_groups[attr_text].append(idx)

print(f"✓ Created {len(attribute_groups)} attribute groups")

# Analyze group sizes
group_sizes = [len(indices) for indices in attribute_groups.values()]
print(f"\n✓ Group size statistics (embeddings only):")
print(f"  Mean:   {np.mean(group_sizes):.1f}")
print(f"  Median: {np.median(group_sizes):.1f}")
print(f"  Min:    {min(group_sizes)}")
print(f"  Max:    {max(group_sizes)}")
print(f"  Groups with 10+ samples:  {sum(1 for s in group_sizes if s >= 10)} ({sum(1 for s in group_sizes if s >= 10)/len(group_sizes)*100:.1f}%)")
print(f"  Groups with 50+ samples:  {sum(1 for s in group_sizes if s >= 50)} ({sum(1 for s in group_sizes if s >= 50)/len(group_sizes)*100:.1f}%)")
print(f"  Groups with 100+ samples: {sum(1 for s in group_sizes if s >= 100)} ({sum(1 for s in group_sizes if s >= 100)/len(group_sizes)*100:.1f}%)")

# Show top groups
print("\n  Top 10 attribute groups:")
sorted_groups = sorted(attribute_groups.items(), key=lambda x: len(x[1]), reverse=True)
for i, (attr_text, indices) in enumerate(sorted_groups[:10]):
    print(f"    {i+1}. '{attr_text}': {len(indices):,} samples")

# ============================================================================
# [7] GPT-4o-mini Augmentation Analysis
# ============================================================================
print("\n[7] GPT-4o-mini Augmentation Impact (10x variants per group)...")

num_text_variants = 10

print(f"\nAssuming {num_text_variants} text variants per attribute group:")
print(f"  Unique attribute groups: {len(attribute_groups)}")
print(f"  Total text descriptions: {len(attribute_groups) * num_text_variants:,}")

# Calculate total training pairs
total_pairs_without_aug = sum(group_sizes)
total_pairs_with_aug = total_pairs_without_aug * num_text_variants

print(f"\n  Training pairs (embedding-text pairs):")
print(f"    Without augmentation: {total_pairs_without_aug:,}")
print(f"    With augmentation:    {total_pairs_with_aug:,}")
print(f"    Increase:             {num_text_variants}x")

# Per-group analysis
print(f"\n  Per-group training pairs (with augmentation):")
aug_group_sizes = [s * num_text_variants for s in group_sizes]
print(f"    Mean:   {np.mean(aug_group_sizes):.1f} pairs/group")
print(f"    Median: {np.median(aug_group_sizes):.1f} pairs/group")
print(f"    Min:    {min(aug_group_sizes)} pairs/group")
print(f"    Max:    {max(aug_group_sizes):,} pairs/group")

# Show augmented counts for top groups
print("\n  Top 10 groups after augmentation:")
sorted_groups_aug = sorted(attribute_groups.items(), key=lambda x: len(x[1]), reverse=True)
for i, (attr_text, indices) in enumerate(sorted_groups_aug[:10]):
    original_count = len(indices)
    augmented_count = original_count * num_text_variants
    print(f"    {i+1}. '{attr_text}':")
    print(f"       {original_count:,} embeddings × {num_text_variants} texts = {augmented_count:,} training pairs")

# Recommendations based on group sizes
print("\n  Recommended GMM components (K) based on group sizes:")
small_groups = sum(1 for s in group_sizes if s < 50)
medium_groups = sum(1 for s in group_sizes if 50 <= s < 200)
large_groups = sum(1 for s in group_sizes if s >= 200)

print(f"    Small groups (<50 embeddings): {small_groups} → Use K=5 components")
print(f"    Medium groups (50-200 embeddings): {medium_groups} → Use K=10 components")
print(f"    Large groups (200+ embeddings): {large_groups} → Use K=15-20 components")

# ============================================================================
# [8] Test Text Encoding (for novel prompts)
# ============================================================================
print("\n[8] Testing text encoding capability...")

try:
    from sentence_transformers import SentenceTransformer

    # Load a small, fast model for testing
    print("  Loading Sentence-BERT model (this may take a moment)...")
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
    print("  ✓ Sentence-BERT loaded successfully")

    # Test encoding
    test_texts = [
        "male, middle-aged adult, medium-pitched, measured speed",
        "female, young adult, high-pitched, fast speed",
        "elderly male with deep voice speaking slowly"
    ]

    embeddings = text_encoder.encode(test_texts)
    print(f"  ✓ Encoded {len(test_texts)} test texts")
    print(f"  ✓ Text embedding dimension: {embeddings.shape[1]}")

    # Test semantic similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    print(f"\n  Semantic similarity matrix:")
    for i, text1 in enumerate(test_texts):
        for j, text2 in enumerate(test_texts):
            if i < j:
                print(f"    '{text1[:40]}...' <-> '{text2[:40]}...': {sim_matrix[i,j]:.3f}")

except ImportError:
    print("  ✗ sentence-transformers not installed")
    print("  Install with: pip install sentence-transformers")
    print("  This is REQUIRED for encoding text prompts")
except Exception as e:
    print(f"  ✗ Error testing text encoding: {e}")

# ============================================================================
# [9] Verify Training Data Availability
# ============================================================================
print("\n[9] Verifying training data availability...")

# Count how many samples we can actually use (already filtered)
usable_samples = len(filtered_indices)
min_group_size = 10  # Minimum samples needed for meaningful GMM

usable_groups = sum(1 for size in group_sizes if size >= min_group_size)

print(f"✓ Usable groups (≥{min_group_size} embeddings): {usable_groups}/{len(attribute_groups)}")
print(f"✓ Usable samples: {usable_samples:,}")

if usable_samples < 50000:
    print("  ⚠ WARNING: Less than 50K usable samples")
elif usable_samples < 100000:
    print("  ✓ Good amount of data for training")
else:
    print("  ✓ Plenty of data for training!")

# ============================================================================
# [10] Sample Training Batch Simulation
# ============================================================================
print("\n[10] Simulating sample training batch...")

# Pick a group with many samples
sample_group = sorted_groups[0]
attr_text, indices = sample_group

print(f"  Using group: '{attr_text}'")
print(f"  Group size: {len(indices)} samples")

# Simulate loading a batch
batch_size = 32
batch_indices = np.random.choice(indices, size=min(batch_size, len(indices)), replace=False)

print(f"\n  Simulated batch of {len(batch_indices)} samples:")
print(f"    Attribute text: '{attr_text}'")
print(f"    Sample audio files:")
for idx in batch_indices[:5]:
    print(f"      - {train_ds[idx]['audio_path']}")

# Check embedding availability for this batch
batch_with_embeddings = sum(
    1 for idx in batch_indices
    if train_ds[idx]['audio_path'] in xvector_lookup
)
print(f"    Embeddings available: {batch_with_embeddings}/{len(batch_indices)}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("READINESS SUMMARY")
print("="*80)

checks = []

# Dataset
checks.append(("Dataset loaded", len(train_ds) > 0))

# Embeddings
checks.append(("Embedding metadata loaded", len(xvector_lookup) > 0))
checks.append(("Data alignment", full_alignment_rate >= 90))

# Libraries (local analysis only)
try:
    from sentence_transformers import SentenceTransformer
    checks.append(("sentence-transformers installed (optional for local)", True))
except:
    checks.append(("sentence-transformers installed (optional for local)", False))

# Data quality
checks.append(("Dataset filtered to sources with embeddings", len(filtered_indices) > 10000))
checks.append(("Sufficient attribute groups", len(attribute_groups) > 50))
checks.append(("Sufficient usable samples", usable_samples > 50000))
checks.append(("Most groups have enough samples", usable_groups/len(attribute_groups) > 0.5 if len(attribute_groups) > 0 else False))

print("\nChecklist:")
for check_name, passed in checks:
    status = "✓" if passed else "✗"
    print(f"  {status} {check_name}")

all_passed = all(passed for _, passed in checks)

if all_passed:
    print("\n" + "="*80)
    print("🎉 ALL CHECKS PASSED - READY FOR GMM-MDN TRAINING!")
    print("="*80)
    print("\nData Summary:")
    print(f"  • Usable samples: {usable_samples:,} (from voxceleb, ears, expresso)")
    print(f"  • Attribute groups: {len(attribute_groups)}")
    print(f"  • With 10x augmentation: {usable_samples * 10:,} training pairs")
    print(f"  • Average: ~{int(np.mean(group_sizes) * 10)} pairs per group")

    print("\nNext steps:")
    print("1. Generate GPT-4o-mini augmented text variants (10 per group)")
    print("2. Create filtered dataset and mapping files")
    print("3. Implement GMM-MDN model architecture on cluster")
    print("4. Create data loader that:")
    print("   - Groups samples by attribute text")
    print("   - Loads embeddings using Hyperion")
    print("   - Returns (attribute_text, [embeddings]) batches")
    print("5. Train model:")
    print("   - Input: attribute text → Text encoder → Dense network")
    print("   - Output: GMM parameters (means, covariances, weights)")
    print("   - Loss: Negative log-likelihood of embeddings under predicted GMM")
    print("6. Evaluate on validation set")
    print("7. Test sampling diverse speakers from same attribute description")
else:
    print("\n" + "="*80)
    print("⚠ SOME CHECKS FAILED - REVIEW ISSUES ABOVE")
    print("="*80)
    print("\nPlease resolve the failed checks before proceeding.")

print("="*80)
