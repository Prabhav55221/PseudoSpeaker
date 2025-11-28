# GMM-MDN Pseudo-Speaker Generation

## Overview

This project implements a Gaussian Mixture Model - Mixture Density Network (GMM-MDN) to generate diverse pseudo-speaker embeddings from text descriptions.

**Problem:** Given a text description of speaker characteristics, generate multiple diverse speaker embeddings that match those characteristics.

**Solution:** Train a neural network to predict GMM parameters from text, then sample from the predicted GMM to get diverse embeddings.

---

## Method

### Architecture

```
Text Description: "male, medium-pitched, measured speed"
    ↓
SentenceBERT Encoder (384-dim)
    ↓
3-Layer Dense Network (512 hidden units)
    ↓
Three Output Heads → GMM Parameters:
    - Means: [K × 512]
    - Log-Variances: [K × 512]
    - Log-Weights: [K]
    ↓
Sample N embeddings from GMM
    ↓
N diverse 512-dim speaker embeddings
```

### Data

**Dataset:** CapSpeech-real (filtered)
- **Total:** 115,948 samples (voxceleb, ears, expresso sources)
- **Splits:** 70% train (~81K) / 10% dev (~12K) / 20% test (~23K)
- **Attribute Groups:** 18 groups (Gender × Pitch × Speaking Rate)
- **Augmentation:** 10 text variants per group via GPT-4o-mini
- **Training Pairs:** 1,159,480 (with 10x augmentation)

**18 Attribute Groups:**
1. male, medium-pitched, measured speed
2. male, low-pitched, measured speed
3. female, high-pitched, measured speed
4. female, medium-pitched, measured speed
5. male, high-pitched, measured speed
6. female, high-pitched, slow speed
7. male, medium-pitched, fast speed
8. male, high-pitched, slow speed
9. male, low-pitched, fast speed
10. male, medium-pitched, slow speed
11. female, medium-pitched, fast speed
12. female, low-pitched, measured speed
13. male, low-pitched, slow speed
14. female, medium-pitched, slow speed
15. female, high-pitched, fast speed
16. female, low-pitched, fast speed
17. male, high-pitched, fast speed
18. female, low-pitched, slow speed

### Training

**Loss:** Negative Log-Likelihood (NLL)
```
L = -mean(log Σᵢ πᵢ·N(x|μᵢ,Σᵢ))
```
Where for each training batch:
- x = target speaker embeddings (from group)
- πᵢ, μᵢ, Σᵢ = predicted GMM parameters

**Optimizer:** AdamW (lr=1e-4, weight_decay=1e-5)
**Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
**Gradient Clipping:** max_norm=1.0

**Training Strategy:**
- Group-based batching: Each batch contains 64 embeddings from ONE attribute group
- Model learns to predict GMM that fits all embeddings in that group
- Text augmentation: Random selection of 1 of 10 text variants per group

### Inference

1. Input text description (e.g., "an elderly man with deep voice")
2. Encode with SentenceBERT
3. Forward through GMM-MDN → Get GMM parameters
4. Sample N embeddings from predicted GMM
5. Use sampled embeddings in TTS system

---

## Implementation Details

### File Structure

```
src/
├── data/
│   ├── prepare_dataset.py    # Create train/dev/test splits
│   ├── dataset.py             # PyTorch Dataset
│   └── dataloader.py          # Custom collation
├── models/
│   ├── text_encoder.py        # SentenceBERT wrapper
│   ├── gmm_mdn.py             # Main GMM-MDN model
│   └── gmm_utils.py           # GMM operations (loss, sampling)
├── training/
│   ├── trainer.py             # Training loop
│   └── evaluator.py           # Validation metrics
└── utils/
    ├── embedding_loader.py    # Hyperion embedding loading
    ├── logger.py              # Logging
    └── config.py              # Configuration

scripts/
├── train.py                   # Main training script
├── sample.py                  # Inference/sampling
└── test_pipeline.py           # End-to-end tester

data_augment/
└── generate_variants.py       # Generate text variants (local)
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Text Encoder | all-MiniLM-L6-v2 | SentenceBERT, frozen |
| Text Embedding Dim | 384 | From SentenceBERT |
| Hidden Dim | 512 | Dense layers |
| GMM Components (K) | 15 | Configurable: 10/15/20 |
| Embedding Dim (D) | 512 | X-vector dimension |
| Batch Size | 64 | Embeddings per group |
| Learning Rate | 1e-4 | AdamW |
| Weight Decay | 1e-5 | L2 regularization |
| Gradient Clip | 1.0 | Max norm |
| Epochs | 100 | With early stopping |

### Evaluation Metrics

1. **NLL (Dev Set):** Primary metric - lower is better
2. **Sampling Diversity:** `mean(std(sampled_embeddings, dim=0))`  - higher is better
3. **Coverage:** % of dev embeddings with high GMM log-prob - higher is better

---

## Usage

### 1. Data Augmentation (Local)

```bash
cd data_augment/
# Create .env with: OPENAI_API_KEY=your_key
python generate_variants.py
# Generates: augmented_texts.json
```

Transfer to cluster:
```bash
scp augmented_texts.json psingh54@login.clsp.jhu.edu:/path/to/project/data_augment/
```

### 2. Data Preparation (Cluster)

```bash
python src/data/prepare_dataset.py \
  --data_dir /export/corpora7/CapSpeech-real/all-real \
  --embedding_dir /path/to/xvectors \
  --output_dir ./prepared_data/mappings
```

Creates:
- `train_mapping.json`
- `dev_mapping.json`
- `test_mapping.json`

### 3. Training

```bash
# Quick test
python scripts/test_pipeline.py --quick

# Full training
bash run.sh
```

Or manually:
```bash
python scripts/train.py \
  --data_dir /export/corpora7/CapSpeech-real/all-real \
  --embedding_dir /path/to/xvectors \
  --augmented_texts ./data_augment/augmented_texts.json \
  --mapping_dir ./prepared_data/mappings \
  --num_gmm_components 15 \
  --batch_size 64 \
  --epochs 100 \
  --output_dir ./outputs/run1
```

### 4. Inference

```bash
python scripts/sample.py \
  --checkpoint ./outputs/run1/best_model.pth \
  --text_description "male, medium-pitched, measured speed" \
  --num_samples 10 \
  --output_file ./outputs/sampled_embeddings.npy
```

---

## Key Design Decisions

### Why GMM-MDN?

1. **One-to-many mapping:** Natural fit for "one description → many speakers"
2. **Interpretable:** GMM components represent different speaker modes
3. **Proven:** TacoSpawn (Google, 2021) showed success with GMM priors
4. **Simple:** Single loss (NLL) vs. CVAE's reconstruction + KL
5. **Fast training:** No encoder-decoder bottleneck

### Why 18 Groups (exclude Age)?

- Validation showed many samples have `age=None`
- Keeping 18 groups ensures all groups have reliable data
- Simpler, more robust for initial implementation
- Can add Age later as optional category

### Why 70/10/20 Split?

- **70% train:** Sufficient data (81K samples with 810K augmented pairs)
- **10% dev:** Enough for validation (12K samples)
- **20% test:** Larger test set for more reliable evaluation

### Why SentenceBERT (frozen)?

- **Pretrained:** Already good at semantic similarity
- **Frozen:** Faster training, less overfitting
- **Small:** all-MiniLM-L6-v2 is fast (384-dim)
- **Upgradeable:** Can fine-tune or use larger model if needed

### Why Group-Based Batching?

- **Signal clarity:** Model sees distribution within each group
- **Efficiency:** Loads embeddings from same group together
- **Effective:** Directly models P(embeddings | group)

---

## Expected Results

### Training

- **Convergence:** 50-100 epochs
- **Time:** 6-12 hours on single GPU
- **Final NLL:** TBD (lower than random baseline)

### Sampling

- **Diversity:** Sampled embeddings should vary (std > 0.1)
- **Quality:** Embeddings should be realistic (coverage > 80%)
- **Consistency:** Sampled embeddings match description attributes

### Deliverables

1. Trained model checkpoint
2. Training logs and loss curves
3. Sample embeddings for all 18 groups
4. Evaluation metrics (NLL, diversity, coverage)

---

## Troubleshooting

### Low Diversity

- Increase GMM components (K=20)
- Reduce batch size (more gradient updates)
- Add diversity regularization

### High NLL (not fitting)

- Increase GMM components
- Add more dense layers
- Increase hidden dim to 768

### Overfitting

- Increase weight decay
- Add dropout (current: 0.1, try 0.2)
- Reduce model capacity

### Text encoder not generalizing

- Fine-tune text encoder
- Try larger model (all-mpnet-base-v2)
- Add more text augmentation variants

---

## Future Extensions

1. **Add Age attribute:** Treat None as "unknown" category
2. **Intrinsic tags:** Include voice qualities (deep, crisp, flowing)
3. **Free-form text:** Support arbitrary descriptions, not just structured attributes
4. **CVAE comparison:** Compare against conditional VAE baseline
5. **TTS integration:** End-to-end evaluation with actual speech synthesis

---

## References

1. **TacoSpawn:** Speaker Generation (Google, 2021) - [Paper](https://google.github.io/tacotron/publications/speaker_generation/)
2. **CapSpeech:** Style-Captioned TTS - [Paper](https://arxiv.org/pdf/2506.02863)
3. **GMM-MDN Prosody:** Phone-Level Modeling - [Paper](https://arxiv.org/abs/2105.13086)
4. **VITS:** Conditional VAE for TTS - [Paper](https://github.com/jaywalnut310/vits)

---

## Contact / Issues

For questions or issues, refer to the main research notes in `docs/CLAUDE.md`
