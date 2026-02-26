#!/usr/bin/env python3
"""
TTS inference using GMM-MDN pseudo-speaker embeddings + IMS-Toucan.

Two modes:

  Mode A — online (single env, no version conflicts):
    Requires avd-hyperion env (has sentence_transformers).
    Samples embeddings from the GMM-MDN on the fly.

    python scripts/tts_inference.py \\
        --checkpoint outputs/.../checkpoints/best_model.pth \\
        --augmented_texts data_augment/augmented_texts.json \\
        --sentences scripts/tts_sentences.txt \\
        --output_dir tts_outputs/ \\
        --toucan_dir /path/to/IMS-Toucan \\
        --num_samples 3

  Mode B — offline (two-stage, avoids huggingface_hub version conflicts):
    Step 1: conda activate avd-hyperion
            python scripts/sample_embeddings.py --checkpoint ... --output_dir emb/
    Step 2: conda activate toucan
            python scripts/tts_inference.py \\
                --embeddings_dir emb/ \\
                --sentences scripts/tts_sentences.txt \\
                --output_dir tts_outputs/ \\
                --toucan_dir /path/to/IMS-Toucan
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize speech using GMM-MDN pseudo-speaker embeddings via IMS-Toucan"
    )

    # ── Source of embeddings (mutually exclusive modes) ───────────────────────
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint",
                     help="[Mode A] Path to best_model.pth checkpoint. "
                          "Requires --augmented_texts. Samples on the fly.")
    src.add_argument("--embeddings_dir",
                     help="[Mode B] Directory produced by sample_embeddings.py "
                          "(contains manifest.json + per-group .npy files). "
                          "No sentence_transformers import — safe in toucan env.")

    # Mode A extras
    parser.add_argument("--augmented_texts",
                        help="[Mode A] Path to augmented_texts.json")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="[Mode A] Embeddings to sample per group (default: 3)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="[Mode A] Sampling temperature (default: 1.0)")
    parser.add_argument("--text_index", type=int, default=0,
                        help="[Mode A] Paraphrase index for GMM condition (default: 0)")

    # Input sentences
    parser.add_argument("--sentences", required=True,
                        help="Path to plain-text file with sentences (one per line)")

    # Output
    parser.add_argument("--output_dir", required=True,
                        help="Directory where WAV files will be saved")

    # IMS-Toucan
    parser.add_argument("--toucan_dir", required=True,
                        help="Path to cloned IMS-Toucan repository root")
    parser.add_argument("--toucan_model", default="Meta",
                        help="IMS-Toucan TTS model (default: Meta)")

    # Group filter (works in both modes)
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Groups to synthesize (default: all). "
                             "Example: 'male, high-pitched, fast speed'")

    # System
    parser.add_argument("--device", default=None,
                        help="Device: cuda or cpu (default: cuda if available)")
    parser.add_argument("-v", "--verbose", action="store_true")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    return logging.getLogger("tts_inference")


def resolve_device(device_arg):
    if device_arg:
        return device_arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_sentences(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sentences file not found: {p}")
    sentences = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    if not sentences:
        raise ValueError(f"No sentences found in {p}")
    return sentences


def build_output_filename(group: str, sample_idx: int, sentence_idx: int) -> str:
    safe_group = group.replace(", ", "_").replace(" ", "_")
    return f"{safe_group}__s{sample_idx:02d}__utt{sentence_idx:02d}.wav"


def import_toucan(toucan_dir: str):
    toucan_path = Path(toucan_dir).resolve()
    if not toucan_path.exists():
        raise FileNotFoundError(f"IMS-Toucan directory not found: {toucan_path}")
    interface_file = toucan_path / "InferenceInterfaces" / "ToucanTTSInterface.py"
    if not interface_file.exists():
        raise FileNotFoundError(
            f"ToucanTTSInterface.py not found at {interface_file}. "
            "Make sure you cloned the full IMS-Toucan repo."
        )
    sys.path.insert(0, str(toucan_path))
    try:
        from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface
        return ToucanTTSInterface
    except ImportError as e:
        raise ImportError(
            f"Failed to import ToucanTTSInterface: {e}\n"
            "Did you run: pip install -r IMS-Toucan/requirements.txt ?"
        ) from e


# ──────────────────────────────────────────────────────────────────────────────
# Mode A — online sampling
# ──────────────────────────────────────────────────────────────────────────────

def load_online_embeddings(checkpoint, augmented_texts, groups, num_samples,
                           temperature, text_index, device, log):
    """
    Load GMM-MDN and sample embeddings on the fly.
    Returns: List of (group_name, sample_idx, embedding_np[192])
    """
    # Lazy import — only reached in Mode A, not when --embeddings_dir is used
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.gmm_mdn import GMMMDN  # noqa: PLC0415

    import torch  # noqa: PLC0415

    log.info(f"Loading GMM-MDN checkpoint: {checkpoint}")
    model, ckpt_meta = GMMMDN.load_checkpoint(checkpoint, device=device)
    model.eval()
    log.info(f"  Epoch: {ckpt_meta.get('epoch', '?')}, dim: {model.embedding_dim}")
    if model.embedding_dim != 192:
        log.warning(f"Embedding dim={model.embedding_dim} (IMS-Toucan expects 192)")

    with open(augmented_texts) as f:
        augmented = json.load(f)

    all_groups = sorted(augmented.keys())
    selected = groups if groups else all_groups
    missing = [g for g in selected if g not in augmented]
    if missing:
        raise ValueError(f"Groups not found in augmented_texts.json: {missing}\n"
                         f"Available: {all_groups}")

    entries = []
    for group in selected:
        variants = augmented[group]
        description = variants[min(text_index, len(variants) - 1)]
        log.info(f"  Sampling {num_samples}x for: {group}")
        for s in range(num_samples):
            with torch.no_grad():
                emb = model.sample(text=description, num_samples=1,
                                   temperature=temperature)
            emb_np = emb.squeeze(0).cpu().numpy()
            log.debug(f"    sample {s}: norm={np.linalg.norm(emb_np):.4f}")
            entries.append((group, s, emb_np))

    return selected, entries


# ──────────────────────────────────────────────────────────────────────────────
# Mode B — offline (pre-computed embeddings from sample_embeddings.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_offline_embeddings(embeddings_dir, groups, log):
    """
    Load embeddings from a directory produced by sample_embeddings.py.
    Returns: (selected_groups, List of (group_name, sample_idx, embedding_np[192]))
    """
    emb_dir = Path(embeddings_dir)
    manifest_path = emb_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {emb_dir}. "
            "Run scripts/sample_embeddings.py first."
        )

    with open(manifest_path) as f:
        manifest_data = json.load(f)

    all_groups = manifest_data["groups"]
    selected = groups if groups else all_groups
    missing = [g for g in selected if g not in all_groups]
    if missing:
        raise ValueError(f"Groups not found in manifest: {missing}\n"
                         f"Available: {all_groups}")

    manifest = manifest_data["manifest"]
    entries = []
    for item in manifest:
        if item["group"] not in selected:
            continue
        npy_path = emb_dir / item["embedding_path"]
        if not npy_path.exists():
            raise FileNotFoundError(f"Embedding file missing: {npy_path}")
        emb_np = np.load(str(npy_path))
        entries.append((item["group"], item["sample_idx"], emb_np))
        log.debug(f"  Loaded {npy_path.name}  norm={np.linalg.norm(emb_np):.4f}")

    log.info(f"Loaded {len(entries)} pre-computed embeddings from {emb_dir}")
    return selected, entries


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    log = setup_logging(args.verbose)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings (one of the two modes) ────────────────────────────────
    if args.embeddings_dir:
        log.info(f"[Mode B] Loading pre-computed embeddings from: {args.embeddings_dir}")
        selected_groups, entries = load_offline_embeddings(
            args.embeddings_dir, args.groups, log
        )
    else:
        if not args.augmented_texts:
            raise ValueError("--augmented_texts is required when using --checkpoint")
        log.info(f"[Mode A] Sampling from checkpoint: {args.checkpoint}")
        selected_groups, entries = load_online_embeddings(
            args.checkpoint, args.augmented_texts, args.groups,
            args.num_samples, args.temperature, args.text_index, device, log
        )

    # ── Load sentences ────────────────────────────────────────────────────────
    sentences = load_sentences(args.sentences)
    log.info(f"Loaded {len(sentences)} sentence(s) from {args.sentences}")

    # ── Load IMS-Toucan ───────────────────────────────────────────────────────
    log.info(f"Importing IMS-Toucan from: {args.toucan_dir}")
    ToucanTTSInterface = import_toucan(args.toucan_dir)

    log.info(f"Initializing IMS-Toucan (model={args.toucan_model}) on {device}…")
    tts = ToucanTTSInterface(device=device, tts_model_path=args.toucan_model)
    log.info("IMS-Toucan ready.")

    # ── Synthesis loop ────────────────────────────────────────────────────────
    total_wavs = len(entries) * len(sentences)
    log.info(f"Synthesizing {total_wavs} WAVs "
             f"({len(entries)} embeddings × {len(sentences)} sentences)")

    wav_count = 0
    for group, sample_idx, spk_emb in entries:
        safe_group = group.replace(", ", "_").replace(" ", "_")
        group_dir = output_dir / safe_group
        group_dir.mkdir(parents=True, exist_ok=True)

        tts.set_utterance_embedding(spk_emb)

        for utt_idx, sentence in enumerate(sentences):
            out_filename = build_output_filename(group, sample_idx, utt_idx)
            out_path = group_dir / out_filename
            log.info(f"  [{wav_count+1}/{total_wavs}] {out_filename}")
            log.debug(f"    Text: \"{sentence}\"")
            tts.read_to_file(text_list=[sentence], file_location=str(out_path))
            wav_count += 1

    log.info(f"\nDone. {wav_count} WAV files saved to: {output_dir}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TTS Inference Summary")
    print("=" * 60)
    print(f"Mode       : {'B (pre-computed)' if args.embeddings_dir else 'A (online)'}")
    print(f"Groups     : {len(selected_groups)}")
    print(f"Embeddings : {len(entries)}")
    print(f"Sentences  : {len(sentences)}")
    print(f"Total WAVs : {wav_count}")
    print(f"Output dir : {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
