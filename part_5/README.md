# Part 5 — Mixture-of-Experts (MoE)

This part focuses purely on the **Mixture-of-Experts feed-forward component**.  
We are **not** implementing self-attention in this module.  
In a full transformer block, the order is generally:

```
[LayerNorm] → [Self-Attention] → [Residual Add]
           → [LayerNorm] → [Feed-Forward Block (Dense or MoE)] → [Residual Add]
```

Our MoE layer is a drop-in replacement for the dense feed-forward block.  
You can imagine it being called **after** the attention output, before the second residual connection.

---
## **5.1 Theory in 60 seconds**
- **Experts**: multiple parallel MLPs; each token activates only a small subset (top-k) → sparse compute.
- **Gate/Router**: scores each token across experts; picks top-k and assigns weights via a softmax.
- **Dispatch/Combine**: send token to chosen experts, run their MLP, combine results using gate weights.
- **Load balancing**: encourage uniform expert usage. A common aux loss (Switch Transformer) is  
  `L_aux = E * Σ ( importance * load )` where:
  - *importance* = avg gate probability per expert  
  - *load* = fraction of tokens routed as primary to that expert
  
---

## **5.3 Distributed notes (single-GPU friendly)**