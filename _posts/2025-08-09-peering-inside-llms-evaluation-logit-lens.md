---
layout: post
title: "Peering Inside LLMs: A small experiment with Logit Lens"
tag: "TOY EXPERIMENTS"
date: 2025-08-09
---


Peering Inside LLMs: A small experiment with Logit Lens
=======================================================

Evaluating and interpreting large language models (LLMs) requires two distinct perspectives. The first is **behavioral evaluation**—treating the model as a black box and analyzing its responses. The second is **mechanistic interpretability**—peering inside the model's activations to understand how it processes inputs.

In this post, we apply the **Logit Lens** (early decoding) to trace how representations evolve in the residual stream of `gpt2-small`.

---

## 1. Structural Interpretation: The Logit Lens

While behavioral evaluation tells us *where* the model fails, **mechanistic interpretability** helps us understand *how* it processes information inside its layers.

In a Transformer model, the **residual stream** acts as a communication channel. We can apply the **Logit Lens** (early decoding) to project intermediate activations back into the vocabulary space by multiplying them with the unembedding matrix W<sub>U</sub>:

<p style="text-align: center; font-family: 'JetBrains Mono', monospace; margin: 1.5em 0;">
  logits<sup>(<em>l</em>)</sup> = h<sup>(<em>l</em>)</sup> &middot; W<sub>U</sub>
</p>

Using `transformer-lens`, we can extract these hidden states and decode them at any layer:

```python
import torch
import transformer_lens

# Load model
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
prompt = "The quick brown fox jumps over the lazy dog"

# Run model and cache intermediate activations
logits, cache = model.run_with_cache(prompt, remove_batch_dim=True)
W_U = model.W_U

# Inspect the residual stream at the second token ("quick")
# post-residual stream state at layer 0
hidden_state = cache['blocks.0.hook_resid_post'][1, :]

# Project to vocabulary
logits_early = hidden_state @ W_U
probs_early = torch.softmax(logits_early, dim=-1)

# Inspect the probability of a target token (e.g., " fox")
target_id = model.to_single_token(" fox")
print("Early decoded probability:", probs_early[target_id].item())
```

---

## 2. Visualizing Token Reconstruction Across Layers

By early-decoding the hidden states at *every* token position and layer, we can inspect how much of the original token identity is preserved versus how much has transitioned into contextual prediction.

Below is a heatmap showing the reconstruction probability of each token in the prompt across all 12 layers of GPT-2:

![Token Reconstruction Heatmap](/assets/img/token_probabilities_heatmap.png)

### Key Takeaways:
1. **Reconstruction in Early Layers**: For content-rich words (like `"brown"`, `"fox"`, `"lazy"`, `"dog"`), the probability remains near `1.00` in layers 0–3. This shows that the initial token embedding dominates the residual stream early on.
2. **Contextual Overwrite**: By layers 4–5, the direct identity of these tokens decays to `0.00` as the residual stream is overwritten with contextual representations needed for predicting the next token.
3. **Grammatical Decoupling**: Functional words (such as `"The"` and `"over"`) drop to `0.00` immediately at Layer 0, indicating they are rapidly contextualized.
