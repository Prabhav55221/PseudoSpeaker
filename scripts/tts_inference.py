#!/usr/bin/env python3
"""
TTS inference using GMM-MDN pseudo-speaker embeddings + IMS-Toucan.

For each attribute group, samples a pseudo-speaker embedding from the
trained GMM-MDN, then synthesizes input sentences using IMS-Toucan,
which natively accepts 192-dim speaker embeddings (same dim as our model).

Usage:
    python scripts/tts_inference.py \\
        --checkpoint outputs/20251128_221541/checkpoints/best_model.pth \\
        --augmented_texts data_augment/augmented_texts.json \\
        --sentences sentences.txt \\
        --output_dir tts_outputs/ \\
        --toucan_dir /path/to/IMS-Toucan \\
        --num_samples 3 \\
        --temperature 1.0

    # Or synthesize for specific groups only:
    python scripts/tts_inference.py \\
        --checkpoint outputs/.../best_model.pth \\
        --augmented_texts data_augment/augmented_texts.json \\
        --sentences sentences.txt \\
        --output_dir tts_outputs/ \\
        --toucan_dir /path/to/IMS-Toucan \\
        --groups "male, high pitch, fast" "female, low pitch, slow"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gmm_mdn import GMMMDN


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthesize speech using GMM-MDN pseudo-speaker embeddings via IMS-Toucan"
    )

    # Model
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pth checkpoint")
    parser.add_argument("--augmented_texts", required=True,
                        help="Path to augmented_texts.json (provides group descriptions)")

    # Input sentences
    parser.add_argument("--sentences", required=True,
                        help="Path to plain-text file with sentences to synthesize (one per line)")

    # Output
    parser.add_argument("--output_dir", required=True,
                        help="Directory where WAV files will be saved")

    # IMS-Toucan location
    parser.add_argument("--toucan_dir", required=True,
                        help="Path to cloned IMS-Toucan repository root")
    parser.add_argument("--toucan_model", default="Meta",
                        help="IMS-Toucan TTS model to use (default: Meta — multilingual)")

    # Group selection
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Attribute groups to synthesize for. Default: all 18 groups. "
                             "Example: 'male, high pitch, fast' 'female, low pitch, slow'")

    # Sampling
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of different speaker embeddings to sample per group (default: 3)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="GMM sampling temperature — higher = more diversity (default: 1.0)")
    parser.add_argument("--text_index", type=int, default=0,
                        help="Which paraphrase index from augmented_texts to use as GMM condition "
                             "(default: 0 — first paraphrase per group)")

    # System
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (default: cuda if available)")
    parser.add_argument("--vocoder", default="default",
                        help="Vocoder to pass to IMS-Toucan (default: IMS-Toucan default)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    return logging.getLogger("tts_inference")


def load_sentences(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sentences file not found: {p}")
    sentences = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    if not sentences:
        raise ValueError(f"No sentences found in {p}")
    return sentences


def load_group_descriptions(augmented_texts_path: str, text_index: int) -> dict:
    """
    Returns {group_name: description_text} using paraphrase at `text_index`.
    group_name is the JSON key (e.g. 'male, high pitch, fast').
    """
    with open(augmented_texts_path) as f:
        augmented = json.load(f)

    descriptions = {}
    for group_name, variants in augmented.items():
        idx = min(text_index, len(variants) - 1)
        descriptions[group_name] = variants[idx]

    return descriptions


def sample_speaker_embedding(
    model: GMMMDN,
    description: str,
    temperature: float,
    device: str,
) -> np.ndarray:
    """
    Sample a single 192-dim speaker embedding from the GMM conditioned on `description`.
    Returns numpy array of shape [192].
    """
    with torch.no_grad():
        sample = model.sample(text=description, num_samples=1, temperature=temperature)
    return sample.squeeze(0).cpu().numpy()  # [192]


def import_toucan(toucan_dir: str):
    """
    Add IMS-Toucan to sys.path and import the TTS interface.
    Returns the ToucanTTSInterface class.

    IMS-Toucan exposes a high-level interface in:
      InferenceInterfaces/ToucanTTSInterface.py

    The class is constructed with:
      ToucanTTSInterface(device, tts_model_path, vocoder_model_path, faster_vocoder)

    Key methods:
      tts.set_utterance_embedding(embedding)  -- set speaker embedding [192] numpy
      tts.read_to_file(texts, path)           -- synthesize list of texts to a single WAV
      tts(text) -> np.ndarray                 -- synthesize single text, return waveform
    """
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


def build_output_filename(group: str, sample_idx: int, sentence_idx: int) -> str:
    """Create a safe filename from group name, sample index, and sentence index."""
    safe_group = group.replace(", ", "_").replace(" ", "_")
    return f"{safe_group}__s{sample_idx:02d}__utt{sentence_idx:02d}.wav"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    log = setup_logging(args.verbose)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load GMM-MDN ──────────────────────────────────────────────────────────
    log.info(f"Loading GMM-MDN checkpoint: {args.checkpoint}")
    model, ckpt_meta = GMMMDN.load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    log.info(
        f"Model loaded — epoch {ckpt_meta.get('epoch', '?')}, "
        f"val_loss {ckpt_meta.get('metrics', {}).get('val_loss', '?'):.4f}"
        if isinstance(ckpt_meta.get('metrics', {}).get('val_loss'), float)
        else f"Model loaded — epoch {ckpt_meta.get('epoch', '?')}"
    )

    embedding_dim = model.embedding_dim
    log.info(f"Embedding dim: {embedding_dim}")
    if embedding_dim != 192:
        log.warning(
            f"Embedding dim is {embedding_dim}, not 192. "
            "IMS-Toucan expects 192-dim speaker embeddings. "
            "Synthesis may produce unexpected results."
        )

    # ── Load group descriptions ────────────────────────────────────────────────
    log.info(f"Loading group descriptions from: {args.augmented_texts}")
    group_descriptions = load_group_descriptions(args.augmented_texts, args.text_index)
    all_group_names = sorted(group_descriptions.keys())
    log.info(f"Found {len(all_group_names)} groups: {all_group_names}")

    # Filter to requested groups
    if args.groups:
        missing = [g for g in args.groups if g not in group_descriptions]
        if missing:
            raise ValueError(
                f"Requested groups not found in augmented_texts.json: {missing}\n"
                f"Available groups: {all_group_names}"
            )
        selected_groups = args.groups
        log.info(f"Using {len(selected_groups)} selected groups")
    else:
        selected_groups = all_group_names
        log.info(f"Using all {len(selected_groups)} groups")

    # ── Load sentences ─────────────────────────────────────────────────────────
    sentences = load_sentences(args.sentences)
    log.info(f"Loaded {len(sentences)} sentence(s) from {args.sentences}")
    for i, s in enumerate(sentences):
        log.debug(f"  [{i}] {s}")

    # ── Load IMS-Toucan ────────────────────────────────────────────────────────
    log.info(f"Importing IMS-Toucan from: {args.toucan_dir}")
    ToucanTTSInterface = import_toucan(args.toucan_dir)

    log.info(f"Initializing IMS-Toucan (model={args.toucan_model}) on {args.device}…")
    tts = ToucanTTSInterface(
        device=args.device,
        tts_model_path=args.toucan_model,
    )
    log.info("IMS-Toucan ready.")

    # ── Synthesis loop ─────────────────────────────────────────────────────────
    total_wavs = len(selected_groups) * args.num_samples * len(sentences)
    log.info(
        f"Synthesizing {total_wavs} WAVs "
        f"({len(selected_groups)} groups × {args.num_samples} samples × {len(sentences)} sentences)"
    )

    wav_count = 0
    for group in selected_groups:
        description = group_descriptions[group]
        group_dir = output_dir / group.replace(", ", "_").replace(" ", "_")
        group_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"\nGroup: {group}")
        log.debug(f"  Description: \"{description}\"")

        for sample_idx in range(args.num_samples):
            # Sample pseudo-speaker embedding from GMM-MDN
            spk_embedding = sample_speaker_embedding(
                model, description, args.temperature, args.device
            )
            log.debug(
                f"  Sample {sample_idx}: embedding norm={np.linalg.norm(spk_embedding):.4f}"
            )

            # Set speaker embedding in IMS-Toucan
            # ToucanTTSInterface.set_utterance_embedding() accepts a 192-dim numpy array
            tts.set_utterance_embedding(spk_embedding)

            for utt_idx, sentence in enumerate(sentences):
                out_filename = build_output_filename(group, sample_idx, utt_idx)
                out_path = group_dir / out_filename

                log.info(f"  [{wav_count+1}/{total_wavs}] {out_filename}")
                log.debug(f"    Text: \"{sentence}\"")

                # Synthesize and save
                # IMS-Toucan's read_to_file writes a WAV to the given path.
                # Pass a list with one sentence so it appends silence-separated utterances.
                tts.read_to_file(
                    text_list=[sentence],
                    file_location=str(out_path),
                )

                wav_count += 1

    log.info(f"\nDone. {wav_count} WAV files saved to: {output_dir}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TTS Inference Summary")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Groups     : {len(selected_groups)}")
    print(f"Samples/grp: {args.num_samples}")
    print(f"Sentences  : {len(sentences)}")
    print(f"Total WAVs : {wav_count}")
    print(f"Output dir : {output_dir.resolve()}")
    print("=" * 60)
    print("\nDirectory layout:")
    for group in selected_groups[:3]:
        safe = group.replace(", ", "_").replace(" ", "_")
        print(f"  {output_dir}/{safe}/")
        for s in range(min(2, args.num_samples)):
            for u in range(min(2, len(sentences))):
                print(f"    {build_output_filename(group, s, u)}")
    if len(selected_groups) > 3:
        print(f"  ... ({len(selected_groups) - 3} more groups)")
    print()


if __name__ == "__main__":
    main()
