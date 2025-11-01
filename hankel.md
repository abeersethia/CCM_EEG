# Hankel Matrix Explained: Complete Guide

## What is a Hankel Matrix?

A **Hankel matrix** is a square or rectangular matrix where each ascending anti-diagonal (top-left to bottom-right) contains the same values. In the context of time series analysis, it's a matrix constructed from a signal where:
- Each **row** represents a time window from the signal
- Rows are **shifted versions** of each other (delay embedding)
- The matrix structure captures **temporal dependencies** in the data

### Mathematical Definition

For a 1D signal `x = [x₁, x₂, x₃, ..., xₙ]`, a Hankel matrix `H` has the form:

```
H = [x₁  x₂  x₃  ... xₘ  ]
    [x₂  x₃  x₄  ... xₘ₊₁]
    [x₃  x₄  x₅  ... xₘ₊₂]
    [ ...               ]
    [xₙ₋ₘ₊₁ ... ... xₙ  ]
```

Each row is a **shifted window** of the original signal.

## Why Use Hankel Matrices?

### 1. **Takens' Embedding Theorem**
The fundamental theorem behind delay embedding states that:
> For a dynamical system with a d-dimensional attractor, a single observable variable can be used to reconstruct the full state space by creating delay-embedded vectors.

**Key insight**: `[X(t), X(t-τ), X(t-2τ), ...]` contains information about the entire system, including hidden dimensions Y and Z!

### 2. **Manifold Reconstruction**
- A Hankel matrix creates a **high-dimensional embedding** from a 1D signal
- This embedding captures the **underlying manifold structure**
- The manifold contains information about the original multi-dimensional system

### 3. **Temporal Structure Preservation**
- Preserves **causal relationships** between time points
- Each row captures a local temporal context
- Overlapping windows maintain continuity

## How It Works in This Project

### 3D Hankel Matrix Structure

In this codebase, we use a **3D Hankel matrix** with the following structure:

```
Shape: (n_batches, delay_embedding_dim, window_len)
```

**Example with parameters:**
- `delay_embedding_dim = 10` (number of delay windows)
- `window_len = 512` (length of each window)
- `stride = 5` (step size between windows)

### Construction Process

#### Step 1: Parameters
```python
signal = [x₁, x₂, x₃, ..., xₙ]  # Original 1D time series
window_len = 512                  # Length of each window
delay_embedding_dim = 10          # Number of overlapping windows per batch
stride = 5                        # Step between consecutive windows
```

#### Step 2: Batch Creation
For each batch `b`, we create `delay_embedding_dim` windows that are shifted by `stride`:

```
Batch 0:
  Window 0: [x₀, x₁, ..., x₅₁₁]           (indices 0:512)
  Window 1: [x₅, x₆, ..., x₅₁₆]           (indices 5:517)
  Window 2: [x₁₀, x₁₁, ..., x₅₂₁]         (indices 10:522)
  ...
  Window 9: [x₄₅, x₄₆, ..., x₅₅₆]         (indices 45:557)

Batch 1:
  Window 0: [x₅, x₆, ..., x₅₁₆]           (indices 5:517)
  Window 1: [x₁₀, x₁₁, ..., x₅₂₁]         (indices 10:522)
  ...
```

#### Step 3: Matrix Formation
The resulting 3D structure:
```
hankel_matrix[batch_idx, delay_idx, :] = signal[start + delay_idx * stride : start + delay_idx * stride + window_len]
```

### Visual Example

For a signal `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]`:
- `window_len = 3`
- `delay_embedding_dim = 4`
- `stride = 1`

**Batch 0:**
```
[[0, 1, 2],     ← Window 0 (indices 0:3)
 [1, 2, 3],     ← Window 1 (indices 1:4)
 [2, 3, 4],     ← Window 2 (indices 2:5)
 [3, 4, 5]]     ← Window 3 (indices 3:6)
```

**Batch 1:**
```
[[1, 2, 3],     ← Window 0 (indices 1:4)
 [2, 3, 4],     ← Window 1 (indices 2:5)
 [3, 4, 5],     ← Window 2 (indices 3:6)
 [4, 5, 6]]     ← Window 3 (indices 4:7)
```

## Implementation Details

### Code Structure

#### 1. Building the Matrix (`build_3d_hankel_matrix`)

```python
def build_3d_hankel_matrix(signal, window_len, delay_embedding_dim, stride=None):
    """
    Creates a 3D Hankel matrix from a 1D signal.
    
    Algorithm:
    1. Calculate how many samples each batch needs
    2. Generate batch start positions
    3. For each batch:
       - Create delay_embedding_dim windows
       - Each window is shifted by stride positions
    """
```

**Key calculations:**
- `samples_per_batch = (delay_embedding_dim - 1) * stride + window_len`
  - Each batch needs enough samples to create all overlapping windows
- `batch_starts = [0, stride, 2*stride, ...]`
  - Starting positions for each batch

#### 2. Dataset Wrapper (`Hankel3DDataset`)

```python
class Hankel3DDataset(Dataset):
    """
    PyTorch Dataset for 3D Hankel matrices.
    
    Features:
    - Normalization (mean/std standardization)
    - Batch shuffling for training
    - Returns batches as tensors
    """
```

**Normalization:**
```python
normalized = (hankel_matrix - mean) / std
```
Important for neural network training (stabilizes gradients).

#### 3. Reconstruction (`reconstruct_from_3d_hankel`)

To reconstruct the original signal from the Hankel matrix:

```python
def reconstruct_from_3d_hankel(hankel_matrix, batch_starts, stride, T):
    """
    Reconstructs signal by averaging overlapping windows.
    
    Algorithm:
    1. For each window in each batch:
       - Add window values to output signal
       - Track how many times each sample was added
    2. Average overlapping regions
    3. Denormalize if needed
    """
```

**Why averaging?** Multiple windows overlap and contain the same signal values. Averaging reduces noise and creates smooth reconstruction.

## Why This Matters for Manifold Reconstruction

### The Pipeline

```
Lorenz Attractor [X, Y, Z]
        ↓
    Extract X only
        ↓
    Hankel Matrix (delay embedding)
        ↓
    Autoencoder
        ↓
    Latent Space ≈ Reconstructed [X, Y, Z]
```

### How Delay Embedding Works

**Takens' Theorem** states that the delay-embedded X component:
```
[X(t), X(t-τ), X(t-2τ), ..., X(t-(m-1)τ)]
```
is **topologically equivalent** to the original state space `[X(t), Y(t), Z(t)]`.

The Hankel matrix creates these delay-embedded vectors:
- Each row is a delay-embedded vector
- Multiple rows capture different time points
- The structure preserves the manifold geometry

### Information Flow

1. **Input**: Only X-component from Lorenz system
2. **Hankel Matrix**: Creates high-dimensional delay embedding
   - Contains information about Y and Z through temporal dependencies
3. **Autoencoder**: Learns to compress and decompress
   - Latent space should capture the 3D manifold structure
4. **Output**: Reconstructed signal (X) or manifold (X, Y, Z via latent space)

## Key Parameters Explained

### `delay_embedding_dim` (e.g., 10)
- **What**: Number of overlapping windows per batch
- **Why**: Determines the embedding dimension for delay reconstruction
- **Trade-off**: 
  - Too small: May not capture full manifold (need at least 2d+1 for d-dimensional attractor)
  - Too large: Increases computational cost and may introduce redundancy

### `window_len` (e.g., 512)
- **What**: Length of each time window
- **Why**: Captures local temporal patterns
- **Trade-off**:
  - Too small: May miss important temporal structure
  - Too large: Increases computation, may introduce irrelevant long-range dependencies

### `stride` (e.g., 5)
- **What**: Step size between consecutive windows
- **Why**: Controls overlap between windows
- **Trade-off**:
  - Small stride (high overlap): More data, smoother reconstruction, higher computation
  - Large stride (low overlap): Less data, faster computation, may miss details

## Practical Example

### Setting Up

```python
from src.core.hankel_matrix_3d import Hankel3DDataset

# Original signal (just X component from Lorenz)
x_signal = lorenz_trajectory[:, 0]  # Shape: (10000,)

# Create Hankel matrix dataset
dataset = Hankel3DDataset(
    signal=x_signal,
    window_len=512,
    delay_embedding_dim=10,
    stride=5,
    normalize=True
)

# Result: dataset.hankel_matrix.shape = (n_batches, 10, 512)
```

### What Gets Created

If `x_signal` has length 10000:
- Number of batches: `(10000 - (10-1)*5 - 512) // 5 + 1 ≈ 1900`
- Each batch: `(10, 512)` → flattened to `(5120,)` for MLP or kept as `(10, 512)` for CNN/LSTM
- Total samples: 1900 batches of delay-embedded windows

### Training

```python
# The model receives:
input_batch = dataset[0]  # Shape: (10, 512)

# Encoder compresses to latent space:
latent = encoder(input_batch)  # Shape: (32, 64) for example

# Decoder reconstructs:
reconstructed = decoder(latent)  # Shape: (1, 512) - reconstructed X window
```

## Mathematical Intuition

### Why It Works

1. **Dynamical Systems**: If you have a system `d/dt [X, Y, Z] = f(X, Y, Z)`, then:
   - X(t) is a function of X(t-τ), Y(t-τ), Z(t-τ) from the past
   - So X(t), X(t-τ), X(t-2τ)... contains information about past Y, Z values
   - The delay embedding reconstructs this relationship

2. **Information Theory**: 
   - The Hankel matrix has high rank if the system is rich in dynamics
   - Low rank indicates redundancy or linear relationships
   - Neural networks can learn the non-linear mapping from delay embedding to full state

3. **Geometry**:
   - The delay-embedded space forms a **manifold**
   - This manifold is **diffeomorphic** (topologically equivalent) to the original attractor
   - The autoencoder learns to compress this manifold while preserving its structure

## Common Issues and Solutions

### Issue 1: Matrix Too Large
**Problem**: Hankel matrix consumes too much memory
**Solution**: 
- Reduce `delay_embedding_dim`
- Increase `stride` (less overlap)
- Use batch processing

### Issue 2: Poor Reconstruction
**Problem**: Model can't reconstruct well from Hankel matrix
**Solutions**:
- Increase `delay_embedding_dim` (may need more embedding dimensions)
- Adjust `window_len` (capture more/fewer temporal patterns)
- Add FNN regularization (ensures proper manifold structure)

### Issue 3: Boundary Effects
**Problem**: Edges of signal have fewer overlapping windows
**Solution**: The `reconstruct_from_3d_hankel` function handles this by averaging only where windows overlap

## Advanced: Connection to Other Methods

### Singular Value Decomposition (SVD)
Hankel matrices are commonly decomposed via SVD:
- Left singular vectors: Temporal patterns
- Singular values: Importance of patterns
- Right singular vectors: Spatial/frequency patterns

### Dynamic Mode Decomposition (DMD)
Hankel matrices are used in DMD to find linear approximations of dynamics.

### System Identification
Hankel matrices are fundamental in identifying state-space models from data.

## Summary

The Hankel matrix in this project:
1. **Transforms** a 1D signal into a structured 3D representation
2. **Implements** delay embedding based on Takens' theorem
3. **Preserves** temporal and causal relationships
4. **Enables** manifold reconstruction from partial observations
5. **Supports** neural network training by creating structured, overlapping windows

The key insight: **A single observed variable (X) contains enough information to reconstruct the full system state (X, Y, Z) when properly delay-embedded into a Hankel matrix structure.**

