# GMM-MDN Pseudo-Speaker Generation

Generate diverse pseudo-speaker embeddings from natural language descriptions using Gaussian Mixture Models and Mixture Density Networks.

## Overview

This project implements a GMM-MDN (Gaussian Mixture Model - Mixture Density Network) system that learns to predict probability distributions over speaker embeddings conditioned on text descriptions. Unlike deterministic models that generate a single embedding per description, this approach models the one-to-many mapping from descriptions to embeddings, enabling diverse pseudo-speaker generation.

**Key Features:**
- Probabilistic modeling of speaker embeddings
- Text-conditioned GMM parameter prediction
- Diverse sample generation from learned distributions
- SentenceBERT text encoder for natural language understanding
- Hyperion integration for x-vector embedding loading

## Architecture

```
Text Description
      ↓
SentenceBERT (frozen)
      ↓
3-Layer Dense Network (MDN Head)
      ↓
GMM Parameters (weights, means, covariances)
      ↓
Sample N diverse speaker embeddings
```

## Project Structure

```
PseudoSpeakers/
├── src/
│   ├── utils/
│   │   ├── config.py              # Configuration dataclass
│   │   ├── logger.py              # Logging utilities
│   │   └── embedding_loader.py    # Hyperion interface for x-vectors
│   ├── models/
│   │   ├── text_encoder.py        # SentenceBERT wrapper
│   │   ├── gmm_utils.py           # GMM operations (NLL, sampling)
│   │   └── gmm_mdn.py             # Main GMM-MDN model
│   ├── data/
│   │   ├── prepare_dataset.py     # Data preparation script
│   │   ├── dataset.py             # PyTorch Dataset
│   │   └── dataloader.py          # Custom collation & batching
│   └── training/
│       ├── trainer.py             # Training loop
│       └── evaluator.py           # Validation metrics
├── scripts/
│   ├── train.py                   # Main training script
│   ├── sample.py                  # Inference script
│   └── test_pipeline.py           # End-to-end tester
├── data_augment/
│   └── generate_variants.py       # GPT-4o-mini text augmentation
├── docs/
│   └── METHOD.md                  # Detailed method documentation
└── run.sh                         # Cluster execution script
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA (for GPU training)
- Hyperion library (for x-vector loading)
- SentenceTransformers

### Installation (Cluster)

The project uses the existing `avd-hyperion` conda environment on the cluster:

```bash
# Clone repository
cd /export/fs06/psingh54/
git clone <repository-url> PseudoSpeakers
cd PseudoSpeakers

# Activate environment
conda activate avd-hyperion

# Install additional dependencies
pip install sentence-transformers tqdm
```

### Data Setup

1. **Mount CapSpeech dataset via sshfs:**

```bash
# On cluster (from login node or compute node)
mkdir -p ~/Documents/JHU/JHUResearch/PseudoSpeakers/data

sshfs username@remote-host:/path/to/CapSpeech-real \
    ~/Documents/JHU/JHUResearch/PseudoSpeakers/data \
    -o allow_other,default_permissions
```

2. **Verify data structure:**

```
data/
├── train.csv
├── val.csv
├── test.csv
└── embeddings/
    └── xvector/
        └── CapSpeech-real/
            ├── *.ark  # Kaldi ARK files
            └── *.csv  # Metadata (audio_id, ark_file, byte_offset)
```

## Usage

### 1. Generate Augmented Text Variants (Local)

Generate diverse natural language descriptions for each attribute group using GPT-4o-mini:

```bash
# Set OpenAI API key
cd data_augment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Generate variants
python generate_variants.py
```

This creates `augmented_texts.json` with 10 text variants per attribute group (180 total).

### 2. Prepare Dataset

Create train/dev/test splits from CapSpeech data:

```bash
python src/data/prepare_dataset.py \
    --data_dir ~/Documents/JHU/JHUResearch/PseudoSpeakers/data \
    --augmented_texts data_augment/augmented_texts.json \
    --output_dir data/mappings \
    --train_ratio 0.7 \
    --dev_ratio 0.1 \
    --seed 42
```

**Output:**
- `data/mappings/train.json` - Training samples
- `data/mappings/dev.json` - Validation samples
- `data/mappings/test.json` - Test samples

**Key features:**
- Splits by speaker (no data leakage)
- Filters to sources with embeddings (voxceleb, ears, expresso)
- 18 attribute groups (Gender × Pitch × Speaking Rate)
- ~1.16M training pairs with 10× augmentation

### 3. Test Pipeline (Optional)

Verify all components work correctly before training:

```bash
python scripts/test_pipeline.py \
    --data_dir ~/Documents/JHU/JHUResearch/PseudoSpeakers/data \
    --embedding_dir ~/Documents/JHU/JHUResearch/PseudoSpeakers/data/embeddings/xvector/CapSpeech-real \
    --mapping_dir data/mappings \
    --device cuda
```

### 4. Training (Cluster)

#### Option A: Submit SLURM Job

```bash
# Edit run.sh to configure paths and hyperparameters
nano run.sh

# Submit job
sbatch run.sh
```

#### Option B: Interactive Training

```bash
# Start interactive GPU session
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=48:00:00 --pty bash

# Activate environment
conda activate avd-hyperion

# Run training
python scripts/train.py \
    --data_dir ~/Documents/JHU/JHUResearch/PseudoSpeakers/data \
    --embedding_dir ~/Documents/JHU/JHUResearch/PseudoSpeakers/data/embeddings/xvector/CapSpeech-real \
    --augmented_texts_path data_augment/augmented_texts.json \
    --mapping_dir data/mappings \
    --num_gmm_components 15 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --freeze_encoder \
    --device cuda \
    --output_dir outputs/run_001
```

**Key hyperparameters:**
- `--num_gmm_components`: Number of GMM components (K) - controls distribution complexity
- `--batch_size`: Embeddings per group per batch
- `--freeze_encoder`: Keep SentenceBERT frozen (recommended)
- `--finetune_encoder`: Fine-tune text encoder (not recommended, increases params)

### 5. Inference / Sampling

Generate pseudo-speaker embeddings from trained model:

```bash
# Single text description
python scripts/sample.py \
    --checkpoint outputs/run_001/checkpoints/best_model.pth \
    --text "A male speaker with deep voice and measured speaking rate" \
    --num_samples 100 \
    --temperature 1.0 \
    --output samples.npy \
    --save_params

# Multiple texts from file
python scripts/sample.py \
    --checkpoint outputs/run_001/checkpoints/best_model.pth \
    --text_file descriptions.txt \
    --num_samples 50 \
    --temperature 1.2 \
    --output samples_batch.npy
```

**Sampling parameters:**
- `--num_samples`: Number of embeddings to generate per text
- `--temperature`: Sampling temperature (higher = more diversity)
  - 1.0: Standard sampling
  - < 1.0: Conservative (less diverse)
  - > 1.0: Aggressive (more diverse)
- `--save_params`: Save GMM parameters to JSON

## Configuration

All hyperparameters are configurable via command-line arguments or `GMMMDNConfig` dataclass:

**Model Architecture:**
- `num_gmm_components`: 15 (default)
- `embedding_dim`: 512 (x-vector dimension, fixed)
- `hidden_dim`: 512 (MDN hidden layer size)
- `text_encoder_name`: "all-MiniLM-L6-v2"

**Training:**
- `batch_size`: 64 (embeddings per group)
- `epochs`: 100
- `lr`: 1e-4
- `weight_decay`: 1e-5
- `grad_clip`: 1.0

**Scheduler & Early Stopping:**
- `scheduler_patience`: 5 epochs
- `scheduler_factor`: 0.5 (LR reduction)
- `early_stopping_patience`: 10 epochs

## Evaluation Metrics

The model is evaluated on:

1. **NLL Loss** (Negative Log-Likelihood) - Primary metric
   - Measures how well the predicted GMM fits ground-truth embeddings
   - Lower is better

2. **Diversity Score** - Average pairwise cosine distance
   - Measures intra-sample diversity
   - Higher is better (more diverse samples)

3. **Coverage Score** - k-NN distance to ground truth
   - Measures how well samples cover the true distribution
   - Lower is better (samples closer to real embeddings)

## Cluster-Specific Instructions

### SLURM Job Configuration

Edit `run.sh` SBATCH directives for your cluster:

```bash
#SBATCH --partition=gpu        # GPU partition name
#SBATCH --gres=gpu:1           # Number of GPUs
#SBATCH --cpus-per-task=8      # CPU cores
#SBATCH --mem=32G              # Memory
#SBATCH --time=48:00:00        # Time limit
```

### Data Mounting

If data is not directly accessible on compute nodes, use sshfs:

```bash
# Mount from login node to compute node
sshfs login-node:/path/to/data ~/data_mount -o allow_other
```

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View logs (live)
tail -f logs/gmm_mdn_<JOB_ID>.out

# View training metrics
tail -f outputs/<run_dir>/train.log

# Cancel job
scancel <JOB_ID>
```

## Troubleshooting

### Common Issues

**1. CUDA out of memory:**
- Reduce `--batch_size`
- Reduce `--num_workers`
- Reduce `--num_gmm_components`

**2. NaN loss during training:**
- Check GMM parameter validation in logs
- Reduce learning rate (`--lr`)
- Check for invalid embeddings

**3. Embedding loading errors:**
- Verify sshfs mount is active
- Check ARK file paths in CSV metadata
- Ensure Hyperion is installed correctly

**4. Import errors:**
- Activate correct conda environment: `conda activate avd-hyperion`
- Install missing dependencies: `pip install sentence-transformers tqdm`

## Citation

If you use this code, please cite:

```bibtex
@misc{pseudo-speakers-gmm-mdn,
  author = {Prabhav Singh},
  title = {GMM-MDN Pseudo-Speaker Generation},
  year = {2024},
  institution = {Johns Hopkins University}
}
```

## License

Apache 2.0

## Contact

Prabhav Singh - psingh54@jhu.edu
