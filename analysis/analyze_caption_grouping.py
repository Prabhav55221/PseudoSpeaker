#!/usr/bin/env python3
"""
Analyze caption grouping strategies for one-to-many learning
This script explores different ways to group similar captions to increase samples per group
"""

import sys
import os
from pathlib import Path
from collections import Counter, defaultdict
from datasets import load_from_disk
import re

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"

print("="*80)
print("Caption Grouping Analysis for One-to-Many Learning")
print("="*80)

# Load dataset
print("\n[1] Loading training dataset...")
try:
    train_ds = load_from_disk(f"{DATA_PATH}/train")
    print(f"✓ Loaded {len(train_ds)} training samples")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Baseline: Current caption distribution
print("\n[2] Baseline: Original Captions")
captions = [ex['caption'] for ex in train_ds]
caption_counts = Counter(captions)
print(f"✓ Unique captions: {len(caption_counts)}")
print(f"✓ Avg samples per caption: {len(train_ds) / len(caption_counts):.2f}")
print(f"✓ Max samples per caption: {max(caption_counts.values())}")
print(f"✓ Captions with 1 sample: {sum(1 for c in caption_counts.values() if c == 1)} ({sum(1 for c in caption_counts.values() if c == 1)/len(caption_counts)*100:.1f}%)")
print(f"✓ Captions with 5+ samples: {sum(1 for c in caption_counts.values() if c >= 5)} ({sum(1 for c in caption_counts.values() if c >= 5)/len(caption_counts)*100:.1f}%)")
print(f"✓ Captions with 10+ samples: {sum(1 for c in caption_counts.values() if c >= 10)} ({sum(1 for c in caption_counts.values() if c >= 10)/len(caption_counts)*100:.1f}%)")

# Strategy 1: Group by structured attributes (gender + age + pitch + speaking_rate)
print("\n[3] Strategy 1: Group by Structured Attributes (Gender+Age+Pitch+Rate)")
attr_groups = defaultdict(list)
for i, ex in enumerate(train_ds):
    key = (
        ex.get('gender', 'unknown'),
        ex.get('age', 'unknown'),
        ex.get('pitch', 'unknown'),
        ex.get('speaking_rate', 'unknown')
    )
    attr_groups[key].append(i)

attr_group_sizes = [len(indices) for indices in attr_groups.values()]
print(f"✓ Unique attribute combinations: {len(attr_groups)}")
print(f"✓ Avg samples per group: {sum(attr_group_sizes) / len(attr_group_sizes):.2f}")
print(f"✓ Max samples per group: {max(attr_group_sizes)}")
print(f"✓ Min samples per group: {min(attr_group_sizes)}")
print(f"✓ Groups with 10+ samples: {sum(1 for s in attr_group_sizes if s >= 10)} ({sum(1 for s in attr_group_sizes if s >= 10)/len(attr_group_sizes)*100:.1f}%)")
print(f"✓ Groups with 50+ samples: {sum(1 for s in attr_group_sizes if s >= 50)} ({sum(1 for s in attr_group_sizes if s >= 50)/len(attr_group_sizes)*100:.1f}%)")
print(f"✓ Groups with 100+ samples: {sum(1 for s in attr_group_sizes if s >= 100)} ({sum(1 for s in attr_group_sizes if s >= 100)/len(attr_group_sizes)*100:.1f}%)")

# Show top 5 groups
print("\n  Top 5 largest attribute groups:")
sorted_groups = sorted(attr_groups.items(), key=lambda x: len(x[1]), reverse=True)
for i, (key, indices) in enumerate(sorted_groups[:5]):
    print(f"    {i+1}. {key}: {len(indices)} samples")

# Strategy 2: Group by basic attributes only (gender + age)
print("\n[4] Strategy 2: Group by Basic Attributes (Gender+Age)")
basic_groups = defaultdict(list)
for i, ex in enumerate(train_ds):
    key = (
        ex.get('gender', 'unknown'),
        ex.get('age', 'unknown')
    )
    basic_groups[key].append(i)

basic_group_sizes = [len(indices) for indices in basic_groups.values()]
print(f"✓ Unique combinations: {len(basic_groups)}")
print(f"✓ Avg samples per group: {sum(basic_group_sizes) / len(basic_group_sizes):.2f}")
print(f"✓ Max samples per group: {max(basic_group_sizes)}")

print("\n  All groups:")
for key, indices in sorted(basic_groups.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"    • {key}: {len(indices)} samples")

# Strategy 3: Group by basic_tags
print("\n[5] Strategy 3: Group by basic_tags (from dataset)")
tag_groups = defaultdict(list)
for i, ex in enumerate(train_ds):
    if ex.get('basic_tags'):
        # Sort tags to create consistent key
        key = tuple(sorted(ex['basic_tags']))
        tag_groups[key].append(i)

tag_group_sizes = [len(indices) for indices in tag_groups.values()]
if tag_group_sizes:
    print(f"✓ Unique tag combinations: {len(tag_groups)}")
    print(f"✓ Avg samples per group: {sum(tag_group_sizes) / len(tag_group_sizes):.2f}")
    print(f"✓ Max samples per group: {max(tag_group_sizes)}")
    print(f"✓ Groups with 10+ samples: {sum(1 for s in tag_group_sizes if s >= 10)} ({sum(1 for s in tag_group_sizes if s >= 10)/len(tag_group_sizes)*100:.1f}%)")
    print(f"✓ Groups with 50+ samples: {sum(1 for s in tag_group_sizes if s >= 50)} ({sum(1 for s in tag_group_sizes if s >= 50)/len(tag_group_sizes)*100:.1f}%)")

    print("\n  Top 10 largest tag groups:")
    sorted_tag_groups = sorted(tag_groups.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (key, indices) in enumerate(sorted_tag_groups[:10]):
        print(f"    {i+1}. {key}: {len(indices)} samples")

# Strategy 4: Extract age/gender from caption text
print("\n[6] Strategy 4: Extract Demographics from Caption Text")
def extract_demographics(caption):
    """Extract age and gender from caption text"""
    caption_lower = caption.lower()

    # Gender
    if 'male' in caption_lower and 'female' not in caption_lower:
        gender = 'male'
    elif 'female' in caption_lower:
        gender = 'female'
    elif 'woman' in caption_lower or "woman's" in caption_lower:
        gender = 'female'
    elif 'man' in caption_lower or "man's" in caption_lower:
        gender = 'male'
    else:
        gender = 'unknown'

    # Age
    if 'elderly' in caption_lower or 'old' in caption_lower:
        age = 'elderly'
    elif 'young' in caption_lower or 'youth' in caption_lower:
        age = 'young'
    elif 'middle-aged' in caption_lower or 'mature' in caption_lower or 'adult' in caption_lower:
        age = 'adult'
    elif 'child' in caption_lower or 'kid' in caption_lower:
        age = 'child'
    else:
        age = 'unknown'

    return (gender, age)

caption_demo_groups = defaultdict(list)
for i, ex in enumerate(train_ds):
    key = extract_demographics(ex['caption'])
    caption_demo_groups[key].append(i)

caption_demo_sizes = [len(indices) for indices in caption_demo_groups.values()]
print(f"✓ Unique demographic groups: {len(caption_demo_groups)}")
print(f"✓ Avg samples per group: {sum(caption_demo_sizes) / len(caption_demo_sizes):.2f}")
print(f"✓ Max samples per group: {max(caption_demo_sizes)}")

print("\n  All caption-extracted demographic groups:")
for key, indices in sorted(caption_demo_groups.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"    • {key}: {len(indices)} samples")

# Strategy 5: Combine intrinsic tags (voice qualities)
print("\n[7] Strategy 5: Group by intrinsic_tags (Voice Qualities)")
intrinsic_groups = defaultdict(list)
for i, ex in enumerate(train_ds):
    if ex.get('intrinsic_tags'):
        # Sort tags to create consistent key
        key = tuple(sorted(ex['intrinsic_tags']))
        intrinsic_groups[key].append(i)

intrinsic_group_sizes = [len(indices) for indices in intrinsic_groups.values()]
if intrinsic_group_sizes:
    print(f"✓ Unique intrinsic tag combinations: {len(intrinsic_groups)}")
    print(f"✓ Avg samples per group: {sum(intrinsic_group_sizes) / len(intrinsic_group_sizes):.2f}")
    print(f"✓ Max samples per group: {max(intrinsic_group_sizes)}")
    print(f"✓ Groups with 10+ samples: {sum(1 for s in intrinsic_group_sizes if s >= 10)} ({sum(1 for s in intrinsic_group_sizes if s >= 10)/len(intrinsic_group_sizes)*100:.1f}%)")
    print(f"✓ Groups with 50+ samples: {sum(1 for s in intrinsic_group_sizes if s >= 50)} ({sum(1 for s in intrinsic_group_sizes if s >= 50)/len(intrinsic_group_sizes)*100:.1f}%)")

    print("\n  Top 10 largest intrinsic tag groups:")
    sorted_intrinsic_groups = sorted(intrinsic_groups.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (key, indices) in enumerate(sorted_intrinsic_groups[:10]):
        tag_str = ', '.join(key[:5])  # Show first 5 tags
        if len(key) > 5:
            tag_str += f', ... ({len(key)} total)'
        print(f"    {i+1}. [{tag_str}]: {len(indices)} samples")

# Summary comparison
print("\n" + "="*80)
print("SUMMARY: Grouping Strategy Comparison")
print("="*80)
print(f"{'Strategy':<40} {'Groups':>10} {'Avg Size':>12} {'Max Size':>12}")
print("-"*80)
print(f"{'Original Captions':<40} {len(caption_counts):>10} {len(train_ds)/len(caption_counts):>12.2f} {max(caption_counts.values()):>12}")
print(f"{'Gender+Age+Pitch+Rate':<40} {len(attr_groups):>10} {sum(attr_group_sizes)/len(attr_group_sizes):>12.2f} {max(attr_group_sizes):>12}")
print(f"{'Gender+Age Only':<40} {len(basic_groups):>10} {sum(basic_group_sizes)/len(basic_group_sizes):>12.2f} {max(basic_group_sizes):>12}")
if tag_group_sizes:
    print(f"{'Basic Tags':<40} {len(tag_groups):>10} {sum(tag_group_sizes)/len(tag_group_sizes):>12.2f} {max(tag_group_sizes):>12}")
print(f"{'Caption-Extracted Demographics':<40} {len(caption_demo_groups):>10} {sum(caption_demo_sizes)/len(caption_demo_sizes):>12.2f} {max(caption_demo_sizes):>12}")
if intrinsic_group_sizes:
    print(f"{'Intrinsic Tags':<40} {len(intrinsic_groups):>10} {sum(intrinsic_group_sizes)/len(intrinsic_group_sizes):>12.2f} {max(intrinsic_group_sizes):>12}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("""
Based on the analysis:

1. BEST FOR TRAINING: Gender+Age grouping
   - Provides good balance: enough groups for diversity, enough samples per group
   - Simple, interpretable grouping
   - Structured attributes are reliable

2. ALTERNATIVE: Gender+Age+Pitch+Rate
   - More fine-grained control
   - Still reasonable samples per group
   - Good for more diverse pseudo-speaker generation

3. FOR DIVERSITY: Intrinsic Tags
   - Captures richer voice qualities (deep, crisp, flowing, etc.)
   - More groups, but may have fewer samples per group

STRATEGY: Start with Gender+Age, train GMM/CVAE, then expand to richer attributes
""")

print("="*80)
