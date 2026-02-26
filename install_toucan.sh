#!/bin/bash
# Install IMS-Toucan and its dependencies for use with tts_inference.py.
#
# Run this once on the cluster (or locally if you have enough RAM/VRAM).
# The Toucan "Meta" model checkpoint (~1 GB) downloads automatically on first run.
#
# Usage:
#   bash install_toucan.sh [optional: target directory]
#
# After installation, use:
#   python scripts/tts_inference.py --toucan_dir ./IMS-Toucan [other args]

set -e

TOUCAN_DIR="${1:-./IMS-Toucan}"

echo "=================================================="
echo "IMS-Toucan Install"
echo "Target directory: $TOUCAN_DIR"
echo "=================================================="

# ── Clone repo ─────────────────────────────────────────────────────────────────
if [ -d "$TOUCAN_DIR" ]; then
    echo "Directory already exists: $TOUCAN_DIR"
    echo "Pulling latest changes..."
    git -C "$TOUCAN_DIR" pull
else
    echo "Cloning IMS-Toucan..."
    git clone https://github.com/DigitalPhonetics/IMS-Toucan.git "$TOUCAN_DIR"
fi

echo ""
echo "Installing Python dependencies..."
pip install -r "$TOUCAN_DIR/requirements.txt"

# IMS-Toucan uses speechbrain for its speaker encoder, which also needs:
pip install speechbrain --quiet

echo ""
echo "=================================================="
echo "Installation complete."
echo ""
echo "Run TTS inference:"
echo ""
echo "  python scripts/tts_inference.py \\"
echo "      --checkpoint outputs/YYYYMMDD_HHMMSS/checkpoints/best_model.pth \\"
echo "      --augmented_texts data_augment/augmented_texts.json \\"
echo "      --sentences scripts/tts_sentences.txt \\"
echo "      --output_dir tts_outputs/ \\"
echo "      --toucan_dir $TOUCAN_DIR \\"
echo "      --num_samples 3 \\"
echo "      --temperature 1.0 \\"
echo "      -v"
echo ""
echo "  # Synthesize specific groups only:"
echo "  python scripts/tts_inference.py \\"
echo "      --checkpoint outputs/.../best_model.pth \\"
echo "      --augmented_texts data_augment/augmented_texts.json \\"
echo "      --sentences scripts/tts_sentences.txt \\"
echo "      --output_dir tts_outputs/ \\"
echo "      --toucan_dir $TOUCAN_DIR \\"
echo "      --groups 'male, high pitch, fast' 'female, low pitch, slow' \\"
echo "      --num_samples 5"
echo ""
echo "  # The first run downloads the Meta TTS model (~1 GB) automatically."
echo "  # Output layout:"
echo "  #   tts_outputs/"
echo "  #     male_high_pitch_fast/"
echo "  #       male_high_pitch_fast__s00__utt00.wav"
echo "  #       male_high_pitch_fast__s00__utt01.wav"
echo "  #       male_high_pitch_fast__s01__utt00.wav"
echo "  #       ..."
echo "  #     female_low_pitch_slow/"
echo "  #       ..."
echo "=================================================="
