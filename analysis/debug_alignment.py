#!/usr/bin/env python3
"""
Debug alignment between dataset audio_path and embedding IDs
"""

import sys
from pathlib import Path
import pandas as pd
from datasets import load_from_disk
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
EMBEDDING_PATH = PROJECT_ROOT / "embeddings"

print("="*80)
print("Debugging Audio-Embedding Alignment")
print("="*80)

# Load dataset
print("\n[1] Loading dataset...")
train_ds = load_from_disk(f"{DATA_PATH}/train")
print(f"✓ Loaded {len(train_ds):,} training samples")

# Load embedding metadata
print("\n[2] Loading embedding metadata...")
csv_files = list(EMBEDDING_PATH.glob("*.csv"))
all_dfs = [pd.read_csv(f) for f in csv_files]
xvector_df = pd.concat(all_dfs, ignore_index=True)
print(f"✓ Loaded {len(xvector_df):,} x-vector entries")

# Get all unique IDs
embedding_ids = set(xvector_df['id'].values)
print(f"✓ Unique embedding IDs: {len(embedding_ids):,}")

# ============================================================================
# Analysis 1: Exact match comparison
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 1: Exact Filename Matching")
print("="*80)

dataset_audio_paths = [ex['audio_path'] for ex in train_ds]
exact_matches = sum(1 for path in dataset_audio_paths if path in embedding_ids)
print(f"Exact matches: {exact_matches:,} / {len(dataset_audio_paths):,} ({exact_matches/len(dataset_audio_paths)*100:.2f}%)")

# ============================================================================
# Analysis 2: Sample comparison
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 2: Sample Filename Comparison")
print("="*80)

print("\nFirst 10 dataset audio_paths:")
for i in range(min(10, len(train_ds))):
    print(f"  {i+1}. {train_ds[i]['audio_path']}")

print("\nFirst 10 embedding IDs:")
for i, emb_id in enumerate(list(embedding_ids)[:10]):
    print(f"  {i+1}. {emb_id}")

# ============================================================================
# Analysis 3: Pattern analysis
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 3: Filename Pattern Analysis")
print("="*80)

# Check if embedding IDs have paths or just filenames
sample_emb_ids = list(embedding_ids)[:100]

has_slash = sum(1 for id in sample_emb_ids if '/' in id)
has_extension = sum(1 for id in sample_emb_ids if id.endswith('.wav'))

print(f"Embedding IDs with '/': {has_slash}/100")
print(f"Embedding IDs with '.wav': {has_extension}/100")

# Check dataset audio_paths
sample_audio_paths = dataset_audio_paths[:100]
dataset_has_slash = sum(1 for path in sample_audio_paths if '/' in path)
dataset_has_wav = sum(1 for path in sample_audio_paths if path.endswith('.wav'))

print(f"Dataset paths with '/': {dataset_has_slash}/100")
print(f"Dataset paths with '.wav': {dataset_has_wav}/100")

# ============================================================================
# Analysis 4: Fuzzy matching attempts
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 4: Fuzzy Matching Strategies")
print("="*80)

# Strategy 1: Strip extensions
print("\n[Strategy 1] Strip .wav extension:")
dataset_stems = [Path(p).stem for p in dataset_audio_paths]
embedding_stems = [Path(id).stem for id in embedding_ids]
embedding_stems_set = set(embedding_stems)

stem_matches = sum(1 for stem in dataset_stems if stem in embedding_stems_set)
print(f"  Matches: {stem_matches:,} / {len(dataset_stems):,} ({stem_matches/len(dataset_stems)*100:.2f}%)")

# Strategy 2: Just filename (no directory)
print("\n[Strategy 2] Just filename (basename):")
dataset_basenames = [Path(p).name for p in dataset_audio_paths]
embedding_basenames = [Path(id).name for id in embedding_ids]
embedding_basenames_set = set(embedding_basenames)

basename_matches = sum(1 for bn in dataset_basenames if bn in embedding_basenames_set)
print(f"  Matches: {basename_matches:,} / {len(dataset_basenames):,} ({basename_matches/len(dataset_basenames)*100:.2f}%)")

# Strategy 3: Stem of basename
print("\n[Strategy 3] Stem of basename (no dir, no extension):")
dataset_basename_stems = [Path(p).stem for p in dataset_basenames]
embedding_basename_stems_set = set([Path(id).stem for id in embedding_ids])

basename_stem_matches = sum(1 for stem in dataset_basename_stems if stem in embedding_basename_stems_set)
print(f"  Matches: {basename_stem_matches:,} / {len(dataset_basename_stems):,} ({basename_stem_matches/len(dataset_basename_stems)*100:.2f}%)")

# ============================================================================
# Analysis 5: Check for prefix/suffix patterns
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 5: Check for Common Prefixes/Patterns")
print("="*80)

# Get common prefixes
dataset_prefixes = Counter([p.split('-')[0] if '-' in p else p.split('_')[0] for p in dataset_basenames])
embedding_prefixes = Counter([id.split('-')[0] if '-' in id else id.split('_')[0] for id in embedding_basenames])

print("\nTop 10 dataset filename prefixes:")
for prefix, count in dataset_prefixes.most_common(10):
    print(f"  {prefix}: {count}")

print("\nTop 10 embedding ID prefixes:")
for prefix, count in embedding_prefixes.most_common(10):
    print(f"  {prefix}: {count}")

# ============================================================================
# Analysis 6: Deep dive into LibriTTS matching
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 6: LibriTTS-R Specific Matching Analysis")
print("="*80)

# Get LibriTTS samples
libritts_samples = [ex for ex in train_ds if ex.get('source') == 'libritts-r']
print(f"LibriTTS-R samples: {len(libritts_samples):,}")

if len(libritts_samples) > 0:
    # Show sample paths
    print("\nSample LibriTTS-R paths (first 10):")
    for i in range(min(10, len(libritts_samples))):
        print(f"  {libritts_samples[i]['audio_path']}")

    # Extract various filename patterns
    sample_libritts = libritts_samples[0]['audio_path']
    # e.g., train-clean-100/103/1241/103_1241_000004_000002.wav

    parts = sample_libritts.split('/')
    print(f"\nPath breakdown for: {sample_libritts}")
    print(f"  Parts: {parts}")
    print(f"  Filename: {parts[-1]}")
    print(f"  Parent dir: {parts[-2] if len(parts) > 1 else 'none'}")

    # Try different matching strategies for LibriTTS
    print("\nTrying LibriTTS-specific matching strategies:")

    # Strategy: Just the numeric ID part (e.g., 103_1241_000004_000002)
    libritts_ids = []
    for ex in libritts_samples[:1000]:
        path = ex['audio_path']
        filename = Path(path).name
        stem = Path(filename).stem
        libritts_ids.append(stem)

    # Check if any embedding IDs contain these patterns
    print(f"\n  Checking if embedding IDs contain LibriTTS patterns...")

    # Get sample LibriTTS ID
    sample_id = libritts_ids[0]
    print(f"  Sample LibriTTS ID: {sample_id}")

    # Search for this pattern in embeddings
    matching_embeddings = [emb_id for emb_id in embedding_ids if sample_id in emb_id]
    if matching_embeddings:
        print(f"  ✓ Found {len(matching_embeddings)} embeddings containing '{sample_id}':")
        for m in matching_embeddings[:5]:
            print(f"    - {m}")
    else:
        print(f"  ✗ No embeddings contain '{sample_id}'")

    # Try searching for just the speaker ID part (103_1241)
    if '_' in sample_id:
        speaker_id = '_'.join(sample_id.split('_')[:2])  # e.g., 103_1241
        print(f"\n  Searching for speaker ID pattern: {speaker_id}")
        speaker_matches = [emb_id for emb_id in list(embedding_ids)[:10000] if speaker_id in emb_id]
        if speaker_matches:
            print(f"  ✓ Found {len(speaker_matches)} embeddings with speaker pattern:")
            for m in speaker_matches[:5]:
                print(f"    - {m}")
        else:
            print(f"  ✗ No embeddings contain speaker pattern '{speaker_id}'")

    # Check if ANY embedding IDs have the numeric pattern XXX_XXXX_...
    print("\n  Checking for XXX_XXXX_... pattern in embeddings:")
    pattern_matches = [emb_id for emb_id in list(embedding_ids) if emb_id[0].isdigit() and '_' in emb_id]
    print(f"  Found {len(pattern_matches)} embedding IDs starting with digit and containing '_'")
    if pattern_matches:
        print(f"  Examples:")
        for m in pattern_matches[:10]:
            print(f"    - {m}")

# ============================================================================
# Analysis 7: Check missing vs. present (updated)
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 7: Examples by Source")
print("="*80)

# Get examples for each source
source_examples = {}
for ex in train_ds:
    source = ex.get('source', 'unknown')
    if source not in source_examples:
        source_examples[source] = {'present': [], 'missing': []}

    audio_path = ex['audio_path']
    if audio_path in embedding_ids:
        if len(source_examples[source]['present']) < 3:
            source_examples[source]['present'].append(audio_path)
    else:
        if len(source_examples[source]['missing']) < 3:
            source_examples[source]['missing'].append(audio_path)

for source, examples in source_examples.items():
    print(f"\n{source}:")
    print("  WITH embeddings:")
    for path in examples['present']:
        print(f"    ✓ {path}")
    print("  WITHOUT embeddings:")
    for path in examples['missing']:
        print(f"    ✗ {path}")

# ============================================================================
# Analysis 8: Dataset Source Field Summary
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 8: Dataset Source Summary & Alignment Rates")
print("="*80)

if 'source' in train_ds.column_names:
    sources = Counter([ex.get('source', 'unknown') for ex in train_ds])
    print("Dataset sources and alignment:")
    for source, count in sources.most_common():
        # Check alignment for full source
        source_samples = [ex for ex in train_ds if ex.get('source') == source]
        source_matches = sum(1 for ex in source_samples if ex['audio_path'] in embedding_ids)
        alignment_rate = source_matches / len(source_samples) * 100 if source_samples else 0

        print(f"\n  {source}: {count:,} samples")
        print(f"    Alignment: {source_matches:,}/{count:,} ({alignment_rate:.2f}%)")
else:
    print("No 'source' field in dataset")

# ============================================================================
# Analysis 9: Check for other potential embedding sources
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 9: Potential Other Embedding Locations")
print("="*80)

print("\nBased on the dataset sources, you might need embeddings from:")
print("  - CapSpeech-real (current - mounted)")
print("  - CapSpeech-MLS (for libritts-r? LibriTTS is part of MLS)")
print("  - Or a separate LibriTTS embedding directory")

print("\nCheck on cluster if these paths exist:")
print("  /home/tthebau1/SHADOW/.../CapSpeech-MLS/")
print("  /home/tthebau1/SHADOW/.../CapSpeech-LibriTTS/")
print("  /home/tthebau1/SHADOW/.../libritts/")

# ============================================================================
# RECOMMENDATION
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

best_strategy = "exact"
best_matches = exact_matches
best_rate = exact_matches/len(dataset_audio_paths)*100

if basename_matches > best_matches:
    best_strategy = "basename"
    best_matches = basename_matches
    best_rate = basename_matches/len(dataset_audio_paths)*100

if stem_matches > best_matches:
    best_strategy = "stem"
    best_matches = stem_matches
    best_rate = stem_matches/len(dataset_audio_paths)*100

if basename_stem_matches > best_matches:
    best_strategy = "basename_stem"
    best_matches = basename_stem_matches
    best_rate = basename_stem_matches/len(dataset_audio_paths)*100

print(f"\nBest matching strategy: {best_strategy}")
print(f"Overall match rate: {best_rate:.2f}% ({best_matches:,}/{len(dataset_audio_paths):,})")

# Check if it's a source-specific issue
if 'source' in train_ds.column_names:
    sources = Counter([ex.get('source', 'unknown') for ex in train_ds])
    libritts_count = sources.get('libritts-r', 0)
    non_libritts_count = len(dataset_audio_paths) - libritts_count

    if libritts_count > 0:
        print(f"\nSource breakdown:")
        print(f"  LibriTTS-R: {libritts_count:,} samples")
        print(f"  Other sources: {non_libritts_count:,} samples")

        # Calculate non-LibriTTS alignment
        if non_libritts_count > 0:
            non_libritts_rate = best_matches / non_libritts_count * 100
            print(f"  Estimated non-LibriTTS alignment: {non_libritts_rate:.1f}%")

            if non_libritts_rate >= 95:
                print("\n✓ DIAGNOSIS: LibriTTS-R embeddings missing, others OK!")
                print("\nRECOMMENDED ACTIONS:")
                print("1. Check on cluster for CapSpeech-MLS embeddings")
                print("2. OR filter dataset to exclude 'libritts-r' source")
                print("3. OR compute LibriTTS embeddings separately")
                print(f"\nYou can still train with {best_matches:,} samples from other sources!")

if best_rate < 50:
    print("\nNext steps:")
    print("1. Check Analysis 6 above for LibriTTS-specific matching attempts")
    print("2. Check on cluster for LibriTTS/MLS embedding directories")
    print("3. Mount additional embedding directories if found")
    print("4. Re-run this script")
    print("5. If not found, proceed with filtering to available sources")

print("="*80)
