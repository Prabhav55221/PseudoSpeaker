#!/usr/bin/env python3
"""
Stage 1: Sample pseudo-speaker embeddings from GMM-MDN and save to disk.

Run this in the avd-hyperion environment (which has sentence_transformers).
The saved embeddings are then consumed by tts_inference.py in the toucan
environment, avoiding any huggingface_hub / transformers version conflicts.

Usage:
    conda activate avd-hyperion
    python scripts/sample_embeddings.py \\
        --checkpoint outputs/.../checkpoints/best_model.pth \\
        --augmented_texts data_augment/augmented_texts.json \\
        --output_dir embeddings_out/ \\
        --num_samples 3 \\
        --temperature 1.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gmm_mdn import GMMMDN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample pseudo-speaker embeddings from GMM-MDN (Stage 1)"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pth checkpoint")
    parser.add_argument("--augmented_texts", required=True,
                        help="Path to augmented_texts.json")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save embeddings (one .npy per group×sample)")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Subset of groups to sample (default: all)")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of speaker embeddings to sample per group (default: 3)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--text_index", type=int, default=0,
                        help="Which paraphrase index to use as GMM condition (default: 0)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    log = logging.getLogger("sample_embeddings")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    log.info(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt_meta = GMMMDN.load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    log.info(f"  Epoch: {ckpt_meta.get('epoch', '?')}, dim: {model.embedding_dim}")

    # Load group descriptions
    with open(args.augmented_texts) as f:
        augmented = json.load(f)

    all_groups = sorted(augmented.keys())
    selected = args.groups if args.groups else all_groups
    missing = [g for g in selected if g not in augmented]
    if missing:
        raise ValueError(f"Groups not found in augmented_texts.json: {missing}")

    log.info(f"Sampling {args.num_samples} embedding(s) for {len(selected)} group(s)...")

    manifest = []  # list of dicts for tts_inference to consume

    for group in selected:
        variants = augmented[group]
        idx = min(args.text_index, len(variants) - 1)
        description = variants[idx]

        safe_group = group.replace(", ", "_").replace(" ", "_")
        group_dir = output_dir / safe_group
        group_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"  Group: {group}")
        log.debug(f"    Condition: \"{description}\"")

        for s in range(args.num_samples):
            with torch.no_grad():
                emb = model.sample(
                    text=description, num_samples=1, temperature=args.temperature
                )
            emb_np = emb.squeeze(0).cpu().numpy()  # [192]

            npy_path = group_dir / f"sample_{s:02d}.npy"
            np.save(str(npy_path), emb_np)

            manifest.append({
                "group": group,
                "sample_idx": s,
                "embedding_path": str(npy_path.relative_to(output_dir)),
                "description": description,
            })
            log.debug(f"    Saved sample {s} → {npy_path.name}  "
                      f"(norm={np.linalg.norm(emb_np):.4f})")

    # Save manifest so tts_inference knows what to load
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"groups": selected, "manifest": manifest}, f, indent=2)

    log.info(f"Saved {len(manifest)} embeddings + manifest → {output_dir}")
    print(f"\nDone. Run TTS stage with:")
    print(f"  conda activate toucan")
    print(f"  python scripts/tts_inference.py \\")
    print(f"      --embeddings_dir {output_dir} \\")
    print(f"      --sentences scripts/tts_sentences.txt \\")
    print(f"      --output_dir tts_outputs/ \\")
    print(f"      --toucan_dir /path/to/IMS-Toucan")


if __name__ == "__main__":
    main()
