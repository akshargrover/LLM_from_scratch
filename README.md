# LLM_from_scratch

## Part 0 — Foundations & Mindset
- **0.1** Understanding the high-level LLM training pipeline (pretraining → finetuning → alignment)
- **0.2** Hardware & software environment setup (PyTorch, CUDA/Mac, mixed precision, profiling tools)

```
conda create -n llm_from_scratch python=3.11
conda activate llm_from_scratch
pip install -r requirements.txt
```

## Part 1 — Core Transformer Architecture
- **1.1** Positional embeddings (absolute learned vs. sinusoidal)
- **1.2** Self-attention from first principles (manual computation with a tiny example)
- **1.3** Building a *single attention head* in PyTorch
- **1.4** Multi-head attention (splitting, concatenation, projections)
- **1.5** Feed-forward networks (MLP layers) — GELU, dimensionality expansion
- **1.5** Residual connections & **LayerNorm**
- **1.6** Stacking into a full Transformer block

## Part 2 — Training a Tiny LLM
- **2.1** Byte-level tokenization
- **2.2** Dataset batching & shifting for next-token prediction
- **2.3** Cross-entropy loss & label shifting
- **2.4** Training loop from scratch (no Trainer API)
- **2.5** Sampling: temperature, top-k, top-p
- **2.6** Evaluating loss on val set

## Part 3 — Modernizing the Architecture
- **3.1** **RMSNorm** (replace LayerNorm, compare gradients & convergence)
- **3.2** **RoPE** (Rotary Positional Embeddings) — theory & code
- **3.3** SwiGLU activations in MLP
- **3.4** KV cache for faster inference
- **3.5** Sliding-window attention & **attention sink**
- **3.6** Rolling buffer KV cache for streaming