#!/usr/bin/env python3
"""
Quick analysis of CapSpeech dataset
Usage: python analyze_capspeech.py
"""

import sys
import os
from pathlib import Path
from collections import Counter
from datasets import load_from_disk
import json

# Paths (using local sshfs mount)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"

print("="*80)
print("CapSpeech Dataset Analysis")
print("="*80)

# Load datasets
print("\n[1] Loading datasets...")
try:
    train_ds = load_from_disk(f"{DATA_PATH}/train")
    val_ds = load_from_disk(f"{DATA_PATH}/val")
    test_ds = load_from_disk(f"{DATA_PATH}/test")
    print(f"✓ Train: {len(train_ds)} samples")
    print(f"✓ Val:   {len(val_ds)} samples")
    print(f"✓ Test:  {len(test_ds)} samples")
    print(f"✓ Total: {len(train_ds) + len(val_ds) + len(test_ds)} samples")
except Exception as e:
    print(f"✗ Error loading datasets: {e}")
    sys.exit(1)

# Analyze captions
print("\n[2] Analyzing captions...")
captions = [ex['caption'] for ex in train_ds]
caption_counts = Counter(captions)
print(f"✓ Unique captions: {len(caption_counts)}")
print(f"✓ Most common caption frequency: {caption_counts.most_common(1)[0][1]}")
print(f"✓ Least common caption frequency: {caption_counts.most_common()[-1][1]}")
print(f"✓ Average samples per caption: {len(train_ds) / len(caption_counts):.2f}")

print("\n  Top 5 most frequent captions:")
for caption, count in caption_counts.most_common(5):
    print(f"    • [{count:4d}x] {caption[:80]}...")

# Analyze attributes
print("\n[3] Analyzing attributes...")
for attr in ['gender', 'age', 'pitch', 'speaking_rate', 'accent']:
    if attr in train_ds.column_names:
        values = [ex[attr] for ex in train_ds if ex[attr]]
        counts = Counter(values)
        print(f"✓ {attr:15s}: {len(counts):3d} unique values")
        print(f"  Top values: {', '.join([f'{v}({c})' for v, c in counts.most_common(3)])}")

# Check tags
print("\n[4] Analyzing tags...")
for tag_type in ['intrinsic_tags', 'situational_tags', 'basic_tags']:
    if tag_type in train_ds.column_names:
        all_tags = []
        for ex in train_ds:
            if ex[tag_type]:
                all_tags.extend(ex[tag_type])
        tag_counts = Counter(all_tags)
        print(f"✓ {tag_type:18s}: {len(tag_counts):3d} unique tags")
        print(f"  Top 5: {', '.join([f'{t}({c})' for t, c in tag_counts.most_common(5)])}")

# Sample examples
print("\n[5] Sample entries:")
for i in range(min(2, len(train_ds))):
    ex = train_ds[i]
    print(f"\n--- Sample {i+1} ---")
    print(f"Audio:   {ex['audio_path']}")
    print(f"Caption: {ex['caption'][:100]}...")
    print(f"Text:    {ex['text'][:80]}...")
    print(f"Gender:  {ex.get('gender', 'N/A')} | Age: {ex.get('age', 'N/A')} | Pitch: {ex.get('pitch', 'N/A')}")
    print(f"Tags:    {ex.get('basic_tags', [])[:3]}")

# Skip embedding check for now
print("\n[6] Embedding analysis...")
print("  (Skipped - embeddings not mounted yet)")

# Duration statistics
print("\n[7] Audio duration statistics...")
durations = [ex['speech_duration'] for ex in train_ds if ex.get('speech_duration')]
if durations:
    print(f"✓ Min duration:  {min(durations):.2f}s")
    print(f"✓ Max duration:  {max(durations):.2f}s")
    print(f"✓ Mean duration: {sum(durations)/len(durations):.2f}s")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
