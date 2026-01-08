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