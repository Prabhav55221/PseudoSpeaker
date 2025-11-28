"""
Prepare dataset for GMM-MDN training.

Creates train/dev/test splits from CapSpeech dataset:
1. Load metadata and filter to sources with embeddings
2. Create attribute groups (Gender × Pitch × Rate)
3. Load augmented text variants
4. Split by speaker (70/10/20) to prevent data leakage
5. Save mappings to JSON files
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd
import argparse
from datasets import load_from_disk


# Sources with available embeddings
VALID_SOURCES = {"voxceleb", "ears", "expresso"}

# Attribute groups (18 total: Gender × Pitch × Rate, excluding Age)
ATTRIBUTE_GROUPS = [
    "male, high-pitched, fast speed",
    "male, high-pitched, measured speed",
    "male, high-pitched, slow speed",
    "male, medium-pitched, fast speed",
    "male, medium-pitched, measured speed",
    "male, medium-pitched, slow speed",
    "male, low-pitched, fast speed",
    "male, low-pitched, measured speed",
    "male, low-pitched, slow speed",
    "female, high-pitched, fast speed",
    "female, high-pitched, measured speed",
    "female, high-pitched, slow speed",
    "female, medium-pitched, fast speed",
    "female, medium-pitched, measured speed",
    "female, medium-pitched, slow speed",
    "female, low-pitched, fast speed",
    "female, low-pitched, measured speed",
    "female, low-pitched, slow speed",
]


def create_attribute_group(row) -> str:
    """
    Create attribute group from structured fields.

    Uses gender, pitch, and speaking_rate fields (excludes age due to None values).
    Dataset fields are already in final format:
    - gender: "male" or "female"
    - pitch: "high-pitched", "medium-pitched", or "low-pitched"
    - speaking_rate: "fast speed", "measured speed", or "slow speed"

    Args:
        row: DataFrame row with gender, pitch, speaking_rate fields

    Returns:
        Attribute group string (e.g., "male, medium-pitched, measured speed")
        Returns None if any required field is missing/invalid
    """
    # Access pandas Series values correctly
    try:
        gender = row['gender'] if pd.notna(row['gender']) else None
        pitch = row['pitch'] if pd.notna(row['pitch']) else None
        speaking_rate = row['speaking_rate'] if pd.notna(row['speaking_rate']) else None
    except:
        return None

    # Validate all required fields are present
    if not gender or not pitch or not speaking_rate:
        return None

    # Values are already in the correct format, just combine them
    attr_group = f"{gender}, {pitch}, {speaking_rate}"

    # Validate it's in our predefined groups
    if attr_group in ATTRIBUTE_GROUPS:
        return attr_group
    else:
        return None


def load_capspeech_metadata(data_dir: Path) -> pd.DataFrame:
    """
    Load and combine CapSpeech metadata from all splits (arrow format).

    Args:
        data_dir: Path to CapSpeech-real directory

    Returns:
        Combined DataFrame with all samples
    """
    dfs = []

    for split in ["train", "val", "test"]:
        split_path = data_dir / split
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split directory: {split_path}")

        # Load arrow dataset
        dataset = load_from_disk(str(split_path))

        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        df["original_split"] = split
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def load_augmented_texts(augmented_texts_path: Path) -> Dict[str, List[str]]:
    """
    Load augmented text variants from JSON.

    Args:
        augmented_texts_path: Path to augmented_texts.json

    Returns:
        Dict mapping attribute_group -> list of text variants
    """
    with open(augmented_texts_path, 'r') as f:
        augmented_texts = json.load(f)

    return augmented_texts


def split_by_speaker(
    samples_by_group: Dict[str, List[Tuple[str, str]]],
    train_ratio: float = 0.7,
    dev_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    Split samples by speaker to prevent data leakage.

    Args:
        samples_by_group: Dict mapping attribute_group -> list of (audio_path, speaker_id)
        train_ratio: Fraction for training
        dev_ratio: Fraction for dev (remaining goes to test)
        seed: Random seed

    Returns:
        Tuple of (train_groups, dev_groups, test_groups)
    """
    random.seed(seed)

    train_groups = defaultdict(list)
    dev_groups = defaultdict(list)
    test_groups = defaultdict(list)

    for attr_group, samples in samples_by_group.items():
        # Group samples by speaker
        speaker_samples = defaultdict(list)
        for audio_path, speaker_id in samples:
            speaker_samples[speaker_id].append(audio_path)

        # Get unique speakers
        speakers = list(speaker_samples.keys())
        random.shuffle(speakers)

        # Split speakers
        num_speakers = len(speakers)
        num_train = int(num_speakers * train_ratio)
        num_dev = int(num_speakers * dev_ratio)

        train_speakers = speakers[:num_train]
        dev_speakers = speakers[num_train:num_train + num_dev]
        test_speakers = speakers[num_train + num_dev:]

        # Assign samples to splits
        for speaker in train_speakers:
            train_groups[attr_group].extend(speaker_samples[speaker])

        for speaker in dev_speakers:
            dev_groups[attr_group].extend(speaker_samples[speaker])

        for speaker in test_speakers:
            test_groups[attr_group].extend(speaker_samples[speaker])

    return dict(train_groups), dict(dev_groups), dict(test_groups)


def create_training_samples(
    split_groups: Dict[str, List[str]],
    augmented_texts: Dict[str, List[str]]
) -> List[Dict]:
    """
    Create training samples with augmented text variants.

    For each audio_path in a group, create N samples (one per text variant).

    Args:
        split_groups: Dict mapping attribute_group -> list of audio_paths
        augmented_texts: Dict mapping attribute_group -> list of text variants

    Returns:
        List of samples: [{"audio_id": ..., "text": ..., "attribute_group": ...}, ...]
    """
    samples = []

    for attr_group, audio_paths in split_groups.items():
        text_variants = augmented_texts.get(attr_group, [])

        if not text_variants:
            print(f"Warning: No text variants for group '{attr_group}', skipping")
            continue

        # Create samples: each audio_path paired with each text variant
        for audio_path in audio_paths:
            for text in text_variants:
                samples.append({
                    "audio_id": audio_path,  # Keep key as audio_id for compatibility
                    "text": text,
                    "attribute_group": attr_group
                })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare GMM-MDN dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to CapSpeech-real directory"
    )
    parser.add_argument(
        "--augmented_texts",
        type=str,
        required=True,
        help="Path to augmented_texts.json"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for train/dev/test JSON files"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.1,
        help="Dev set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    augmented_texts_path = Path(args.augmented_texts)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Dataset Preparation for GMM-MDN")
    print("=" * 80)

    # Load CapSpeech metadata
    print("\n1. Loading CapSpeech metadata...")
    df = load_capspeech_metadata(data_dir)
    print(f"   Total samples: {len(df):,}")

    # Filter to valid sources
    print("\n2. Filtering to sources with embeddings...")
    # Check if source field exists, otherwise derive from audio_path
    if "source" not in df.columns:
        df["source"] = df["audio_path"].str.split("_").str[0]
    df_filtered = df[df["source"].isin(VALID_SOURCES)].copy()
    print(f"   Filtered samples: {len(df_filtered):,}")
    print(f"   Sources: {sorted(df_filtered['source'].unique())}")

    # Create attribute groups from structured fields
    print("\n3. Creating attribute groups from structured fields...")
    print(f"   DEBUG: DataFrame columns: {list(df_filtered.columns)}")
    print(f"   DEBUG: First row sample:")
    if len(df_filtered) > 0:
        first_row = df_filtered.iloc[0]
        for col in ['gender', 'pitch', 'speaking_rate', 'age', 'audio_path']:
            if col in df_filtered.columns:
                print(f"     {col}: {first_row[col]}")
            else:
                print(f"     {col}: [COLUMN NOT FOUND]")

    df_filtered["attribute_group"] = df_filtered.apply(create_attribute_group, axis=1)

    print(f"   DEBUG: Sample attribute_group values: {df_filtered['attribute_group'].head(10).tolist()}")

    df_filtered = df_filtered[df_filtered["attribute_group"].notna()].copy()
    print(f"   Samples with valid attributes: {len(df_filtered):,}")
    print(f"   Attribute groups: {df_filtered['attribute_group'].nunique()}")

    # Check for speaker_id field or derive it
    if "speaker_id" not in df_filtered.columns:
        print("   Note: speaker_id field not found, deriving from audio_path")
        # Extract speaker ID from audio_path (format: source_speakerID_utteranceID)
        df_filtered["speaker_id"] = df_filtered["audio_path"].str.split("_").str[1]

    # Group samples by attribute group
    samples_by_group = defaultdict(list)
    for _, row in df_filtered.iterrows():
        audio_path = row["audio_path"]
        speaker_id = row["speaker_id"]
        attr_group = row["attribute_group"]
        samples_by_group[attr_group].append((audio_path, speaker_id))

    # Show group statistics
    print("\n   Samples per group:")
    for group in sorted(samples_by_group.keys()):
        count = len(samples_by_group[group])
        print(f"     {group:45s}: {count:6,}")

    # Load augmented texts
    print("\n4. Loading augmented text variants...")
    augmented_texts = load_augmented_texts(augmented_texts_path)
    print(f"   Loaded variants for {len(augmented_texts)} groups")
    variants_per_group = [len(v) for v in augmented_texts.values()]
    print(f"   Variants per group: {min(variants_per_group)}-{max(variants_per_group)}")

    # Split by speaker
    print("\n5. Splitting by speaker...")
    print(f"   Ratios: {args.train_ratio:.1%} train / {args.dev_ratio:.1%} dev / {1 - args.train_ratio - args.dev_ratio:.1%} test")
    train_groups, dev_groups, test_groups = split_by_speaker(
        samples_by_group,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed
    )

    train_count = sum(len(v) for v in train_groups.values())
    dev_count = sum(len(v) for v in dev_groups.values())
    test_count = sum(len(v) for v in test_groups.values())

    print(f"   Train: {train_count:,} samples")
    print(f"   Dev:   {dev_count:,} samples")
    print(f"   Test:  {test_count:,} samples")

    # Create training samples with augmented texts
    print("\n6. Creating training samples with text variants...")
    train_samples = create_training_samples(train_groups, augmented_texts)
    dev_samples = create_training_samples(dev_groups, augmented_texts)
    test_samples = create_training_samples(test_groups, augmented_texts)

    print(f"   Train: {len(train_samples):,} samples (with augmentation)")
    print(f"   Dev:   {len(dev_samples):,} samples (with augmentation)")
    print(f"   Test:  {len(test_samples):,} samples (with augmentation)")

    # Save to JSON
    print("\n7. Saving to JSON files...")
    train_path = output_dir / "train.json"
    dev_path = output_dir / "dev.json"
    test_path = output_dir / "test.json"

    with open(train_path, 'w') as f:
        json.dump(train_samples, f, indent=2)
    print(f"   Saved: {train_path}")

    with open(dev_path, 'w') as f:
        json.dump(dev_samples, f, indent=2)
    print(f"   Saved: {dev_path}")

    with open(test_path, 'w') as f:
        json.dump(test_samples, f, indent=2)
    print(f"   Saved: {test_path}")

    print("\n" + "=" * 80)
    print("Dataset preparation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
