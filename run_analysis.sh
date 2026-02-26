#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=gmm_analysis

#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --mail-user="psingh54@jhu.edu"

# GMM-MDN Post-Training Analysis Script
#
# Usage:
#   sbatch run_analysis.sh
#
# Or for interactive testing:
#   bash run_analysis.sh

set -e  # Exit on error

echo "=================================================="
echo "GMM-MDN Post-Training Analysis"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=================================================="

# ============================================================================
# Configuration
# ============================================================================

# Project directory
PROJECT_DIR="/export/fs06/psingh54/PseudoSpeaker"
cd "$PROJECT_DIR"

# Model checkpoint (update this to your trained model)
CHECKPOINT="$PROJECT_DIR/outputs/20260210_121035/checkpoints/checkpoint_epoch_66.pth"

# Data paths
EMBEDDING_DIR="/home/tthebau1/SHADOW/iarpa-arts/recipes/voxceleb_eval/v3.6.xs/exp/xvectors/fbank80_stmn_fwseresnet34.v3.1_arts_srevox.s2/CapSpeech-real"
AUGMENTED_TEXTS="$PROJECT_DIR/data_augment/augmented_texts.json"
MAPPING_DIR="$PROJECT_DIR/data/mappings"

# Output directory with timestamp
OUTPUT_DIR="$PROJECT_DIR/outputs/analysis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Analysis parameters
MAX_PER_GROUP=600       # Max real embeddings per group
NUM_GENERATED=600       # Generated embeddings per group for visualization
NUM_MC_SAMPLES=10000    # Monte Carlo samples for KL/JSD
TEMPERATURE=1.0         # Sampling temperature
DEVICE="cuda"
SEED=42

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

# ============================================================================
# Verify Paths
# ============================================================================

echo ""
echo "Verifying paths..."

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -d "$EMBEDDING_DIR" ]; then
    echo "ERROR: Embedding directory not found: $EMBEDDING_DIR"
    exit 1
fi

if [ ! -f "$AUGMENTED_TEXTS" ]; then
    echo "ERROR: Augmented texts file not found: $AUGMENTED_TEXTS"
    exit 1
fi

if [ ! -f "$MAPPING_DIR/test.json" ]; then
    echo "ERROR: test.json not found in: $MAPPING_DIR"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo "Embedding dir: $EMBEDDING_DIR"
echo "Augmented texts: $AUGMENTED_TEXTS"
echo "Mapping dir: $MAPPING_DIR"
echo "Output dir: $OUTPUT_DIR"

# ============================================================================
# Run Analysis
# ============================================================================

echo ""
echo "=================================================="
echo "Starting Analysis"
echo "=================================================="
echo "Analyses: 2.5 (visualization), 2.6 (GMM comparison), 2.7 (classification), 2.1 (KL divergence)"
echo "Max per group: $MAX_PER_GROUP"
echo "Num generated: $NUM_GENERATED"
echo "MC samples: $NUM_MC_SAMPLES"
echo "Temperature: $TEMPERATURE"
echo "=================================================="

python scripts/analyze_model.py \
    --checkpoint "$CHECKPOINT" \
    --embedding_dir "$EMBEDDING_DIR" \
    --mapping_dir "$MAPPING_DIR" \
    --augmented_texts "$AUGMENTED_TEXTS" \
    --output_dir "$OUTPUT_DIR" \
    --analyses 2.5 2.6 2.7 2.1 \
    --max_per_group "$MAX_PER_GROUP" \
    --num_generated "$NUM_GENERATED" \
    --num_mc_samples "$NUM_MC_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    -vv

# ============================================================================
# Post-Analysis
# ============================================================================

echo ""
echo "=================================================="
echo "Analysis Complete!"
echo "=================================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR/"
echo ""
echo "Plots:"
ls -la "$OUTPUT_DIR/plots/" 2>/dev/null || echo "No plots directory"
echo ""
echo "Report: $OUTPUT_DIR/report.md"
echo "Results JSON: $OUTPUT_DIR/results.json"
echo "=================================================="
echo "Job finished: $(date)"
