# Pseudo Speaker Generation Research

## Project Overview

**Goal:** Generate diverse pseudo-speaker embeddings from text descriptions using probabilistic models.

**Input:** Text description (e.g., "a female elderly person with deep voice")
**Output:** Probability distribution (GMM) over speaker embeddings → sample N diverse speakers

## Available Data

### CapSpeech Datasets
- **Paper:** https://arxiv.org/pdf/2506.02863
- **Audio Locations:**
  - Real: `/export/corpora7/CapSpeech-real/audios`
  - CommonVoice: `/export/corpora7/CapSpeech-CommonVoice/commonvoice`
  - GigaSpeech: `/export/corpora7/CapSpeech-GigaSpeech/gigaspeech`
  - MLS: `/export/corpora7/CapSpeech-MLS/mls_english`

### Pre-computed Speaker Embeddings (X-vectors)
- CommonVoice: `/home/tthebau1/SHADOW/iarpa-arts/recipes/voxceleb_eval/v3.6.xs/exp/xvectors/fbank80_stmn_fwseresnet34.v3.1_arts_srevox.s2/CapSpeech-CommonVoice/`
- Real: `/home/tthebau1/SHADOW/iarpa-arts/recipes/voxceleb_eval/v3.6.xs/exp/xvectors/fbank80_stmn_fwseresnet34.v3.1_arts_srevox.s2/CapSpeech-real/`
- MLS: `/home/tthebau1/SHADOW/iarpa-arts/recipes/voxceleb_eval/v3.6.xs/exp/xvectors/fbank80_stmn_fwseresnet34.v3.1_arts_srevox.s2/CapSpeech-MLS/`
- GigaSpeech: `/home/tthebau1/SHADOW/iarpa-arts/recipes/voxceleb_eval/v3.6.xs/exp/xvectors/fbank80_stmn_fwseresnet34.v3.1_arts_srevox.s2/CapSpeech-GigaSpeech/`

## Current Challenge

- Existing datasets have 1-to-1 mapping: description → single speaker embedding
- Need: description → distribution of embeddings
- Question: How to learn probabilistic mapping when training data is deterministic?

---

## Research Notes

### CapSpeech Paper Analysis

CapSpeech uses **style-captioned text-to-speech** with natural language captions:

**Speaker Attributes Annotated:**
- Vocal qualities: age, gender, accent, emotion
- Speech style: pitch, speaking rate, voice timbre
- Prosodic features: intonation patterns, emotional expression
- Demographic traits: speaker identity

**Models Used:**
- LLMs: Mistral-7B for caption generation
- Speech encoders: Whisper for transcription
- Speaker modeling: Age-gender recognition networks
- Acoustic analysis: Pitch extraction and prosodic features
- Text encoders: BERT-based models

**Key Insight:** Natural language captions are paired with audio and speaker embeddings, enabling flexible TTS conditioning.

### Related Work

#### 1. TacoSpawn (Google, 2021)
**Approach:** Fits a GMM over Tacotron2 speaker embeddings to learn a prior distribution over speakers
**Architecture:** Dense ReLU network with one-hot encoded (gender, age-group) labels parameterizing a 10-16 component GMM
**Limitation:** Trained to maximize likelihood of speaker embeddings rather than data likelihood, limiting representational capability
**Reference:** [Speaker Generation](https://google.github.io/tacotron/publications/speaker_generation/)

#### 2. VoiceLens (2023)
**Approach:** Controllable speaker generation and editing with normalizing flows
**Key Innovation:** Uses flow-based models for controllable speaker embedding generation
**Reference:** [VoiceLens](https://arxiv.org/abs/2309.14094)

#### 3. Cross-Utterance Conditioned VAE (CUC-VAE, 2024)
**Approach:** Extracts acoustic, speaker, and textual features from surrounding sentences
**Strength:** Enhances prosody and natural speech generation using pre-trained LMs
**Reference:** [CUC-VAE](https://arxiv.org/abs/2309.04156)

#### 4. VITS (Conditional VAE)
**Approach:** End-to-end TTS with CVAE formulation using variational inference and normalizing flows
**Key Insight:** Addresses one-to-many mapping problem in TTS (text → diverse speech realizations)
**Reference:** [VITS](https://github.com/jaywalnut310/vits)

#### 5. GMM-Based Prosody Modeling (Phone-Level)
**Approach:** Uses GMM-based mixture density network (MDN) to predict GMM distribution of phone-level prosody embeddings
**Innovation:** Random sampling from predicted GMM during inference for diverse speech generation
**Extension:** Supports multi-speaker TTS
**Reference:** [Phone-Level Prosody Modelling](https://arxiv.org/abs/2105.13086)

#### 6. VoxGenesis (2024)
**Approach:** Unsupervised discovery of latent speaker manifold
**Relevance:** Learns continuous speaker representations without explicit labels
**Reference:** [VoxGenesis](https://arxiv.org/abs/2403.00529)

#### 7. Semantic-VAE (2025)
**Approach:** Semantic alignment regularization in latent space to overcome reconstruction-generation trade-off
**Performance:** 2.10% WER, 0.64 speaker similarity on LibriSpeech
**Reference:** [Semantic-VAE](https://arxiv.org/abs/2509.22167)

#### 8. Rethinking Speaker Embeddings (2024)
**Approach:** Multiple sub-centers per speaker class during training to capture intra-speaker diversity
**Innovation:** Single speaker can have multiple embedding sub-centers capturing variation
**Reference:** [Sub-Center Modeling](https://arxiv.org/abs/2407.04291)

### Key Insights from Literature

1. **One-to-Many Problem:** Text-to-speech inherently has one-to-many mapping (one text → many valid speech realizations)
2. **VAE/CVAE Success:** Variational autoencoders effective for modeling speech diversity
3. **GMM Prior:** Gaussian mixture models successfully used as priors over speaker embeddings (TacoSpawn)
4. **Intra-Speaker Diversity:** Even single speakers exhibit diversity (sub-center modeling)
5. **Zero-Shot Capability:** Modern systems can sample novel speakers by interpolating/sampling from learned distributions

---

## Proposed Methodologies

### Method 1: Text-Conditioned GMM-MDN (Mixture Density Network)

**Inspiration:** Combination of TacoSpawn + Phone-Level GMM Prosody + Text conditioning

**Architecture:**
```
Text Description (e.g., "elderly female with deep voice")
    ↓
Text Encoder (BERT/Sentence-BERT)
    ↓
Dense Network (ReLU layers)
    ↓
GMM Parameters (means, covariances, mixture weights)
    ↓
Sample from GMM → Pseudo-Speaker Embeddings
```

**Training Strategy:**
1. **Data Preparation:**
   - Group CapSpeech samples by semantic similarity of descriptions
   - For each description cluster, collect all corresponding speaker embeddings
   - This creates a "one description → many embeddings" training set

2. **Model Training:**
   - Input: Text description embedding
   - Output: Parameters of K-component GMM (K=10-16)
   - Loss: Negative log-likelihood of true speaker embeddings under predicted GMM
   - Formula: `Loss = -log(Σᵢ πᵢ·N(x|μᵢ,Σᵢ))`

3. **Inference:**
   - Input text description → Predict GMM parameters
   - Sample N diverse speaker embeddings from GMM
   - Use embeddings with TTS system

**Pros:**
- Directly inspired by proven TacoSpawn approach
- Simple and interpretable
- Explicitly models distribution as mixture of Gaussians
- Easy to sample diverse speakers

**Cons:**
- Requires semantic clustering of descriptions (may lose information)
- GMM may be too simple for complex embedding distributions
- Fixed number of mixture components

---

### Method 2: Conditional Variational Autoencoder (CVAE)

**Inspiration:** VITS + CUC-VAE + One-to-many TTS literature

**Architecture:**
```
Speaker Embedding (x) + Text Description (c)
    ↓
Encoder: q(z|x,c) → μ, σ (posterior)
Prior: p(z|c) → μ_prior, σ_prior
    ↓
Latent Space z ~ N(μ, σ)
    ↓
Decoder: p(x|z,c) → Reconstructed Embedding
```

**Training Strategy:**
1. **Standard CVAE Loss:**
   ```
   L = Reconstruction_Loss + β·KL_Divergence
   L = ||x - x̂||² + β·KL(q(z|x,c) || p(z|c))
   ```

2. **Key Innovation:**
   - Encoder learns: Given (embedding, description) → latent representation
   - Prior network learns: Given description → distribution over latent space
   - This captures the **many valid embeddings per description**

3. **Data Augmentation Strategy:**
   - Use multiple speaker embeddings per description (from different datasets)
   - Apply perturbations to embeddings to increase diversity
   - Use paraphrasing to augment text descriptions

4. **Inference:**
   - Input: Text description only
   - Sample z ~ p(z|c) from learned prior
   - Decode: x = Decoder(z, c)
   - Repeat N times for N diverse speakers

**Pros:**
- Theoretically grounded (variational inference)
- Naturally handles one-to-many mapping
- Can learn complex distributions
- β parameter controls diversity vs. quality trade-off

**Cons:**
- More complex than GMM approach
- Requires careful tuning of β (KL weight)
- May suffer from posterior collapse
- Need sufficient diversity in training data

---

### Method 3: Hierarchical VAE with Sub-Center Clustering

**Inspiration:** "Rethinking Speaker Embeddings" + Hierarchical generative modeling

**Key Idea:** Model intra-description diversity explicitly with hierarchical structure

**Architecture:**
```
Text Description (c)
    ↓
Description-Level Latent: z_desc ~ p(z_desc|c)
    ↓
Speaker-Level Latent: z_spk ~ p(z_spk|z_desc)
    ↓
Speaker Embedding: x ~ p(x|z_spk, z_desc)
```

**Training Strategy:**
1. **Hierarchical Structure:**
   - First level: Capture high-level description semantics (gender, age, etc.)
   - Second level: Capture within-description variability

2. **Sub-Center Learning:**
   - For each description, learn K sub-centers (K=3-5)
   - Each sub-center represents a different "type" of speaker fitting the description
   - Use mixture of sub-centers as prior

3. **Training Objective:**
   ```
   L = Reconstruction + KL(q(z_spk|x) || p(z_spk|z_desc)) + KL(q(z_desc|x) || p(z_desc|c))
   ```

4. **Inference:**
   - Sample z_desc ~ p(z_desc|c)
   - Sample z_spk ~ p(z_spk|z_desc)  [K times from K sub-centers]
   - Decode: x = Decoder(z_spk, z_desc)

**Pros:**
- Explicitly models intra-description diversity
- Hierarchical structure captures multi-scale variation
- Sub-centers provide interpretable diversity modes
- Can generate very diverse speakers

**Cons:**
- Most complex architecture
- Requires more training data
- Harder to tune and train
- Multiple levels of KL divergence to balance

---

### Method 4: Normalizing Flow-Based Generator (VoiceLens-Inspired)

**Inspiration:** VoiceLens + Flow-based generative models

**Key Idea:** Learn invertible transformations between simple base distribution and speaker embedding space, conditioned on text

**Architecture:**
```
Text Description (c) → Text Encoder
    ↓
z ~ N(0, I) [Base distribution]
    ↓
Flow Transformations: f₁, f₂, ..., fₙ (each conditioned on c)
    ↓
x = fₙ(...f₂(f₁(z, c), c)..., c)
```

**Training Strategy:**
1. **Flow Training:**
   - Learn invertible transformations f that map N(0,I) → speaker embedding distribution
   - Condition each flow layer on text description
   - Train by maximizing likelihood: `log p(x|c) = log p(f⁻¹(x)) + log|det(∂f⁻¹/∂x)|`

2. **Advantages of Flows:**
   - Exact likelihood computation (no KL approximation like VAE)
   - Exact sampling (invertible transformations)
   - Can model complex distributions

3. **Inference:**
   - Sample z ~ N(0, I)
   - Apply forward flow: x = f(z, c)
   - Repeat N times for N diverse speakers

**Pros:**
- Exact likelihood and sampling
- No posterior collapse issues
- Very expressive (can model complex distributions)
- State-of-art in image generation

**Cons:**
- Computationally expensive
- Requires careful architecture design (invertibility constraints)
- Less research in speaker embedding generation specifically
- May be overkill for this task

---

### Method 5: Hybrid: GMM Post-Processing on VAE Latent Space

**Inspiration:** Combine simplicity of GMM with expressiveness of VAE

**Key Idea:** Use VAE to learn good latent representation, then fit GMM in latent space

**Architecture:**
```
Stage 1 (Pre-training):
    Speaker Embedding → VAE → Latent Space z
    (Unsupervised embedding compression)

Stage 2 (Text-Conditioned GMM):
    Text Description → Dense Network → GMM parameters in latent space z
    Sample z ~ GMM → VAE Decoder → Speaker Embedding
```

**Training Strategy:**
1. **Stage 1:** Train standard VAE on all speaker embeddings (no text conditioning)
   - This learns a good compressed representation of speaker space
   - Latent space z should be lower dimensional and better behaved

2. **Stage 2:** Train text-conditioned GMM-MDN in learned latent space
   - Input: Text description
   - Output: GMM parameters (μ, Σ, π) in z-space
   - Loss: NLL of true latent codes under predicted GMM

3. **Inference:**
   - Text → GMM parameters in z-space
   - Sample z ~ GMM
   - Decode: x = VAE_Decoder(z)

**Pros:**
- Best of both worlds: VAE learns representation, GMM provides interpretability
- Two-stage training is simpler to debug
- GMM in latent space may work better than in raw embedding space
- Can pre-train VAE on larger unlabeled data

**Cons:**
- Two-stage training (less end-to-end)
- VAE pre-training may not optimize for text-conditioning task
- Requires tuning for both stages

---

## Recommended Approach Ranking

### For Initial Experiments:

**1. Method 5 (Hybrid VAE + GMM)** - RECOMMENDED FOR FIRST TRY
- Good balance of simplicity and expressiveness
- Two-stage training easier to debug
- Can leverage existing VAE implementations
- GMM provides interpretability

**2. Method 1 (Text-Conditioned GMM-MDN)** - GOOD BASELINE
- Simplest approach, proven by TacoSpawn
- Fast to implement and train
- Good starting point to establish feasibility
- Easy to interpret and debug

**3. Method 2 (CVAE)** - STRONG THEORETICAL CHOICE
- Well-studied in literature
- Natural formulation for one-to-many problem
- More expressive than pure GMM

### For Advanced Experiments (if initial methods work):

**4. Method 3 (Hierarchical VAE)** - If you need more diversity
**5. Method 4 (Normalizing Flows)** - If you need state-of-art quality

---

## Implementation Plan

### Phase 1: Data Exploration & Preparation (Week 1)
1. Explore CapSpeech dataset structure
   - Load and inspect text descriptions
   - Load corresponding speaker embeddings (x-vectors)
   - Analyze embedding dimensions and statistics
   - Visualize embedding space (t-SNE/UMAP)

2. Create description clustering
   - Compute text embedding similarity
   - Group similar descriptions together
   - Analyze cluster sizes and diversity
   - Create train/val/test splits

3. Data statistics and feasibility check
   - How many unique descriptions?
   - How many embeddings per description?
   - Distribution of characteristics (gender, age, etc.)

### Phase 2: Baseline Implementation (Week 2-3)
1. Implement Method 1 (GMM-MDN baseline)
   - Simple text encoder (BERT/SentenceBERT)
   - Dense network predicting GMM parameters
   - Train and evaluate

2. Evaluation metrics
   - **Diversity:** Measure spread of sampled embeddings
   - **Correctness:** Do embeddings match description? (use pretrained classifiers)
   - **Quality:** Use in TTS system, measure naturalness
   - **Coverage:** How well does GMM cover true embedding distribution?

3. Baseline results and analysis

### Phase 3: Advanced Methods (Week 4-6)
1. Implement Method 5 (Hybrid VAE+GMM)
2. Implement Method 2 (CVAE)
3. Compare all methods quantitatively and qualitatively

### Phase 4: Integration & Testing (Week 7-8)
1. Integrate best model with TTS system
2. Large-scale sampling experiments
3. Human evaluation of generated speech
4. Paper writing and documentation

---

## Evaluation Metrics

### 1. Diversity Metrics
- **Intra-Description Variance:** Measure spread of sampled embeddings for same description
- **Inter-Description Distance:** Embeddings from different descriptions should be far apart
- **Coverage:** Compare distribution of sampled embeddings to distribution of real embeddings (KL divergence, Wasserstein distance)

### 2. Correctness Metrics
- **Attribute Classification:** Use pre-trained classifiers (gender, age, emotion) to verify sampled embeddings match description
- **Embedding Similarity:** Cosine similarity between sampled and real embeddings for same description

### 3. Quality Metrics (When Used in TTS)
- **MOS (Mean Opinion Score):** Human evaluation of naturalness
- **Speaker Similarity:** Perceptual evaluation - does generated speech match description?
- **WER (Word Error Rate):** Run ASR on generated speech

### 4. Distribution Metrics
- **Negative Log-Likelihood:** How well does predicted distribution fit held-out embeddings?
- **FID (Fréchet Inception Distance):** Adapted for speaker embeddings
- **Precision/Recall:** Precision (quality) vs Recall (diversity) trade-off

---

## Technical Requirements

### Software Stack
- **Python 3.8+**
- **PyTorch 2.0+**
- **Transformers (HuggingFace):** For text encoders
- **Kaldi/SpeechBrain:** For x-vector handling (if needed)
- **NumPy, SciPy:** For GMM computations
- **Scikit-learn:** For clustering and metrics
- **Matplotlib, Seaborn:** Visualization
- **Weights & Biases / TensorBoard:** Experiment tracking

### Compute Requirements
- **Training:** 1-2 GPUs (V100 or A100)
- **Embedding dimension:** Likely 256-512D (x-vectors)
- **Batch size:** 64-128
- **Training time:** 1-3 days per method

### Data Requirements
- Need to verify exact format of x-vectors in provided paths
- May need to create mapping file: description → embedding paths
- Estimate: 10K-100K description-embedding pairs (depends on dataset)

---

## Next Steps

1. **Immediate:** Explore the dataset on cluster
   - SSH to cluster
   - Navigate to data paths
   - Understand file formats
   - Count samples
   - Load sample embeddings

2. **Data Processing Script:** Create script to:
   - Load text descriptions
   - Load corresponding x-vectors
   - Create clean dataset
   - Compute statistics

3. **Implement Baseline:** Start with Method 1 (GMM-MDN)
   - Fastest to implement
   - Validates feasibility
   - Establishes evaluation pipeline

4. **Iterate:** Based on baseline results, decide next steps
