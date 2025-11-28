#!/usr/bin/env python3
"""
Inference script for GMM-MDN pseudo-speaker generation.

Generate pseudo-speaker embeddings from text descriptions.

Usage:
    python scripts/sample.py --checkpoint /path/to/model.pth --text "A male speaker with deep voice"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gmm_mdn import GMMMDN


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate pseudo-speaker embeddings from text"
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )

    # Text input (one of these required)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text",
        type=str,
        help="Text description (single sample)"
    )
    text_group.add_argument(
        "--text_file",
        type=str,
        help="Path to file with text descriptions (one per line)"
    )

    # Sampling parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate per text (default: 1)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more diversity) (default: 1.0)"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="samples.npy",
        help="Output file path (.npy) (default: samples.npy)"
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu) (default: cuda if available)"
    )

    # Additional options
    parser.add_argument(
        "--save_params",
        action="store_true",
        help="Save GMM parameters (weights, means, vars) to JSON"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Verbosity level: (default INFO), -v for INFO, -vv for DEBUG"
    )

    return parser.parse_args()


def load_texts(args):
    """
    Load text descriptions from args.

    Returns:
        List of text strings
    """
    if args.text:
        return [args.text]
    elif args.text_file:
        text_file = Path(args.text_file)
        if not text_file.exists():
            raise FileNotFoundError(f"Text file not found: {text_file}")

        with open(text_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]

        return texts
    else:
        raise ValueError("Must provide either --text or --text_file")


def main():
    args = parse_args()

    print("=" * 80)
    print("GMM-MDN Pseudo-Speaker Generation - Inference")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, checkpoint = GMMMDN.load_checkpoint(args.checkpoint, device=args.device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        print(f"  Validation loss: {checkpoint['metrics'].get('val_loss', 'unknown')}")

    print(f"\n{model}")

    # Load text descriptions
    texts = load_texts(args)
    print(f"\nLoaded {len(texts)} text description(s)")

    # Generate samples
    print(f"\nGenerating {args.num_samples} sample(s) per text (temperature={args.temperature})...")

    all_samples = []
    gmm_params = []

    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] Text: \"{text}\"")

        # Generate samples
        samples = model.sample(
            text=text,
            num_samples=args.num_samples,
            temperature=args.temperature
        )

        samples_np = samples.cpu().numpy()
        all_samples.append(samples_np)

        print(f"  Generated {len(samples_np)} embeddings (shape: {samples_np.shape})")

        # Optionally save GMM parameters
        if args.save_params:
            weights, means, log_vars = model.get_gmm_params(text)

            params = {
                "text": text,
                "weights": weights.cpu().numpy().tolist(),
                "means": means.cpu().numpy().tolist(),
                "log_vars": log_vars.cpu().numpy().tolist()
            }
            gmm_params.append(params)

    # Save samples
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(texts) == 1:
        # Single text: save as [num_samples, 512]
        np.save(output_path, all_samples[0])
    else:
        # Multiple texts: save as list
        np.save(output_path, all_samples)

    print(f"\nSaved samples to: {output_path}")

    # Save GMM parameters if requested
    if args.save_params:
        params_path = output_path.with_suffix('.params.json')
        with open(params_path, 'w') as f:
            json.dump(gmm_params, f, indent=2)
        print(f"Saved GMM parameters to: {params_path}")

    # Print statistics
    print("\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)

    for i, (text, samples_np) in enumerate(zip(texts, all_samples)):
        print(f"\nText {i+1}: \"{text}\"")
        print(f"  Shape: {samples_np.shape}")
        print(f"  Mean norm: {np.linalg.norm(samples_np, axis=1).mean():.4f}")
        print(f"  Std norm: {np.linalg.norm(samples_np, axis=1).std():.4f}")

        if samples_np.shape[0] > 1:
            # Compute pairwise cosine similarities
            samples_norm = samples_np / np.linalg.norm(samples_np, axis=1, keepdims=True)
            similarities = np.dot(samples_norm, samples_norm.T)
            # Average off-diagonal similarities
            mask = ~np.eye(similarities.shape[0], dtype=bool)
            avg_sim = similarities[mask].mean()
            print(f"  Avg pairwise similarity: {avg_sim:.4f}")
            print(f"  Avg pairwise distance: {1 - avg_sim:.4f}")

    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
