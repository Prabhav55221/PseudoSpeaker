#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=gmm_mdn
#SBATCH --output=logs/gmm_mdn_%j.out
#SBATCH --error=logs/gmm_mdn_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user="psingh54@jhu.edu"

# GMM-MDN Training Script for Cluster Execution
#
# Usage:
#   sbatch run.sh
#
# Or for interactive testing:
#   bash run.sh

set -e  # Exit on error

echo "=================================================="
echo "GMM-MDN Pseudo-Speaker Generation - Cluster Training"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=================================================="

# ============================================================================
# Configuration
# ============================================================================

# Project directory
PROJECT_DIR="/export/fs06/psingh54/PseudoSpeakers"  # UPDATE if needed
cd "$PROJECT_DIR"

# Data paths (UPDATE THESE if needed)
DATA_DIR="$HOME/Documents/JHU/JHUResearch/PseudoSpeakers/data"  # Mounted via sshfs
EMBEDDING_DIR="$DATA_DIR/embeddings/xvector/CapSpeech-real"
AUGMENTED_TEXTS="$PROJECT_DIR/data_augment/augmented_texts.json"
MAPPING_DIR="$PROJECT_DIR/data/mappings"

# Output directory
OUTPUT_DIR="$PROJECT_DIR/outputs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Training hyperparameters
NUM_GMM_COMPONENTS=15
BATCH_SIZE=64
EPOCHS=100
LR=1e-4
HIDDEN_DIM=512

# System settings
NUM_WORKERS=8
DEVICE="cuda"

# ============================================================================
# Setup Environment
# ============================================================================

echo ""
echo "Setting up environment..."

# Load modules
source /home/psingh54/.bashrc
module load cuda/11.7

# Activate conda environment
conda activate avd-hyperion

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Verify CUDA availability
echo ""
echo "Checking CUDA..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install additional dependencies if needed
echo ""
echo "Installing additional dependencies..."
pip install sentence-transformers tqdm --quiet

# ============================================================================
# Verify Data
# ============================================================================

echo ""
echo "Verifying data paths..."

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Please mount data using sshfs or update DATA_DIR path"
    exit 1
fi

if [ ! -d "$EMBEDDING_DIR" ]; then
    echo "ERROR: Embedding directory not found: $EMBEDDING_DIR"
    exit 1
fi

if [ ! -f "$AUGMENTED_TEXTS" ]; then
    echo "ERROR: Augmented texts file not found: $AUGMENTED_TEXTS"
    echo "Please run: python data_augment/generate_variants.py"
    exit 1
fi

echo "✓ Data paths verified"

# ============================================================================
# Prepare Dataset (if not already done)
# ============================================================================

if [ ! -d "$MAPPING_DIR" ] || [ ! -f "$MAPPING_DIR/train.json" ]; then
    echo ""
    echo "Preparing dataset (creating train/dev/test splits)..."

    python src/data/prepare_dataset.py \
        --data_dir "$DATA_DIR" \
        --augmented_texts "$AUGMENTED_TEXTS" \
        --output_dir "$MAPPING_DIR" \
        --train_ratio 0.7 \
        --dev_ratio 0.1 \
        --seed 42

    echo "✓ Dataset prepared"
else
    echo "✓ Using existing dataset mappings from: $MAPPING_DIR"
fi

# ============================================================================
# Test Pipeline (optional, comment out for production runs)
# ============================================================================

# Uncomment to test pipeline before training
# echo ""
# echo "Testing pipeline..."
# python scripts/test_pipeline.py \
#     --data_dir "$DATA_DIR" \
#     --embedding_dir "$EMBEDDING_DIR" \
#     --mapping_dir "$MAPPING_DIR" \
#     --device "$DEVICE"

# ============================================================================
# Training
# ============================================================================

echo ""
echo "=================================================="
echo "Starting Training"
echo "=================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Number of GMM components: $NUM_GMM_COMPONENTS"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "=================================================="

python scripts/train.py \
    --data_dir "$DATA_DIR" \
    --embedding_dir "$EMBEDDING_DIR" \
    --augmented_texts_path "$AUGMENTED_TEXTS" \
    --mapping_dir "$MAPPING_DIR" \
    --num_gmm_components "$NUM_GMM_COMPONENTS" \
    --hidden_dim "$HIDDEN_DIM" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --freeze_encoder \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --log_interval 100 \
    --save_interval 5 \
    --scheduler_patience 5 \
    --scheduler_factor 0.5 \
    --early_stopping_patience 10 \
    --seed 42

# ============================================================================
# Post-training
# ============================================================================

echo ""
echo "=================================================="
echo "Training Complete!"
echo "=================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Best model: $OUTPUT_DIR/checkpoints/best_model.pth"
echo "Logs: $OUTPUT_DIR/train.log"
echo "=================================================="

# Optional: Test sampling
echo ""
echo "Testing sampling..."
python scripts/sample.py \
    --checkpoint "$OUTPUT_DIR/checkpoints/best_model.pth" \
    --text "A male speaker with deep voice and measured speaking rate" \
    --num_samples 10 \
    --temperature 1.0 \
    --output "$OUTPUT_DIR/test_samples.npy" \
    --save_params

echo ""
echo "Job finished: $(date)"