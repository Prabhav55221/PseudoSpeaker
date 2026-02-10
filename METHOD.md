# Pseudo-Speaker Generation with GMM-MDN

## Data

### CapSpeech

The dataset includes recordings from VoxCeleb, EARS, and Expresso sources (only these three have pre-extracted x-vector embeddings). Total size is approximately 1.2M samples.

Each audio file has structured metadata:
- Gender: male or female
- Pitch: high-pitched, medium-pitched, or low-pitched
- Speaking rate: fast speed, measured speed, or slow speed

This gives us 18 attribute combinations (2 × 3 × 3).

### Text Descriptions

Ysed GPT-4o-mini to generate 10 natural language paraphrases for each of the 18 attribute groups. This gives us 180 diverse text descriptions.

For example, "male, medium-pitched, measured speed" becomes:
- "A man with a medium tone and an even tempo in his speech"
- "A male voice characterized by a moderate pitch and a controlled pace"
- "A gentleman speaking with a balanced voice and a deliberate speed"

## Model Architecture

### Text Encoder

SentenceBERT (all-MiniLM-L6-v2) to encode text descriptions into 384-dimensional embeddings. The encoder has 22M parameters and is frozen during training. We only train the MDN network on top.

### Mixture Density Network

The MDN takes the 384-dim text embedding and predicts parameters for a Gaussian Mixture Model with K=15 components in 192-dimensional space.

Architecture:
1. Three dense layers (384→512→512→512) with ReLU activations and dropout
2. Three output heads:
   - Mixture weights: predicts 15 values, softmax normalized
   - Component means: predicts 15 × 192 values, reshaped to [15, 192]
   - Component log-variances: predicts 15 × 192 values, reshaped to [15, 192]

Each GMM component is a 192-dim Gaussian with diagonal covariance. The mixture weights determine how likely each component is to generate a sample.

## Training

### Loss Function

Negative log-likelihood (NLL) loss. For each text description, we compute the probability of the real x-vectors under the predicted GMM. The model learns to predict a GMM that assigns high probability to all real speakers with those attributes.

For a batch with text T and x-vectors {x₁, x₂, ..., xₙ}:
1. Predict GMM parameters (weights, means, log-vars) from T
2. Compute log-probability of each xᵢ under the GMM
3. Average the negative log-probabilities

This encourages the model to spread the GMM components to cover all the diverse real speakers.

## Inference

At inference, we provide a text description like "A female speaker with high-pitched voice and fast speaking rate." The model:

1. Encodes the text to 384-dim using SentenceBERT
2. Predicts GMM parameters (15 components in 192-dim space)
3. Samples x-vectors from the GMM using temperature-controlled sampling

Trained model achieves:
- Diversity: 13.25 
- Coverage: 0.95 
- Mahalanobis: 1.82 

## Additions:

1. Test with single or double combinations of attributes
   1. Only male speaker or like 2 out of three (Done)
   2. Genralized training - think about batching
2. Metrics
   1. KL Divergence between real and predicted GMM. (Done) - See other distance metrics.
   2. TTS - test against real TTS.
   3. Balance between quality and diversity.
   4. Test the attributes like gender and stuff like speaker rate etc.
   5. Embedding space visualization. - See if embeddings are close. 
   6. See if the GMMs for each prompts are different from each other. Also union/intersection of them to see if they work - Like cancelling out of opposite attributes.
   7. See if it goes in the reverse direction - Test if embedding goes back to the correct prompt based on the GMM. (Done)
   
3. Learnable projection to 512 embeddings for TTS.
4. Classification loss for the different categories?
   1. Finetuning for balancing the classes


Generate multiple embeddings and then select from them. 