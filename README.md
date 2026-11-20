# Layer-Wise Representation Geometry and Contextual Disambiguation in Decoder-Only Transformers

**A Comparative Mechanistic Study of `Pythia-410M`, `Qwen2.5-1.5B`, and `DeepSeek-R1-Distill-Qwen-1.5B`**

---

## Abstract

While static token embedding tables ($W_E \in \mathbb{R}^{V \times d}$) map discrete token IDs to deterministic vectors, contextual meaning and opaque "black-box" behavior emerge across subsequent residual stream transformations. In this study, we investigate how representation dimensionality and linear feature separability co-evolve across transformer layers during contextual processing. Using a controlled probe suite scaled to $N = 100$ evaluation samples, we measure **Effective Rank** (via singular value entropy) alongside **Linear Probing Accuracy** and **Sub-Block Attribution** ($\Delta_{\text{Attn}}$ vs. $\Delta_{\text{MLP}}$) across three open-weight models spanning distinct architectural and post-training regimes: `EleutherAI/pythia-410m`, `Qwen/Qwen2.5-1.5B`, and `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

Our results show that global representation breadth (Effective Rank) expands rapidly over the first $4\text{--}6$ layers and stabilizes onto a high-dimensional plateau ($\approx 75\text{--}84$) across all three models, whereas linear separability of contextual/sense features follows markedly different trajectories depending on architecture and reasoning distillation. In particular, `DeepSeek-R1-Distill-Qwen-1.5B` exhibits a pronounced mid-early linear probing trough around Layer 8 ($\approx 0.63$) despite maintaining high effective rank ($\approx 81$), alongside stronger final-layer MLP shifts ($\Delta_{\text{MLP}} = 0.281$ vs. $0.243$ in base `Qwen2.5-1.5B`).

---

## 1. Motivation & Research Questions

In autoregressive transformers, an input token $t$ begins at Layer 0 as a context-free lookup vector:

$$x_t^{(0)} = e_{\text{id}_t}^\top W_E \in \mathbb{R}^d$$

Through $L$ residual blocks, multi-head self-attention (MHSA) and feed-forward networks (MLP) iteratively update this working representation:

$$x_t^{(\ell+1)} = x_t^{(\ell)} + a_t^{(\ell)} + m_t^{(\ell)}$$

where $a_t^{(\ell)}$ and $m_t^{(\ell)}$ denote the attention and MLP sub-block outputs at layer $\ell$. Rather than treating the model solely via end-to-end next-token logits, this work addresses three mechanistic questions:

1. **Dimensional Expansion:** How rapidly do token representations expand from a narrower initial embedding subspace into a higher-dimensional contextual manifold across layers?
2. **Breadth vs. Linear Readability:** Does higher effective rank imply cleaner linear separability of semantic/syntactic distinctions, or can representations remain high-dimensional while temporarily obscuring linear readout?
3. **Effect of Reasoning Distillation:** Holding tokenizer and dense architecture constant (`Qwen2.5-1.5B` vs. `DeepSeek-R1-Distill-Qwen-1.5B`), how does reasoning-trace distillation alter layer-wise feature organization and late-layer sub-block contributions?

---

## 2. Evaluated Models

To isolate architectural and post-training effects without gated-weight barriers, we evaluate three open-weight causal language models:

| Model ID | Parameters | Architecture Details | Role in Study |
| :--- | :---: | :--- | :--- |
| `EleutherAI/pythia-410m` | 410M | 24 layers, GELU, untied $W_E / W_U$, parallel Attn+MLP | Classic interpretability baseline |
| `Qwen/Qwen2.5-1.5B` | 1.5B | 28 layers, RoPE, GQA, SwiGLU, sequential Attn $\rightarrow$ MLP | Modern dense base transformer |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | Same dense 28-layer backbone as `Qwen2.5-1.5B` | Reasoning-distilled comparison target |

---

## 3. Methodology & Metrics

### 3.1 Activation Extraction
For each prompt in our controlled evaluation suite, we cache residual-stream states $x_t^{(\ell)}$ and sub-block updates ($a_t^{(\ell)}, m_t^{(\ell)}$) at the target token position across all layers $\ell \in \{0, \dots, L-1\}$.

### 3.2 Effective Rank (Spectral Dimensionality)
To quantify how broadly variance is distributed across orthogonal directions at layer $\ell$, we center the activation matrix $X^{(\ell)} \in \mathbb{R}^{N \times d}$ across samples ($\tilde{X}^{(\ell)} = X^{(\ell)} - \bar{X}^{(\ell)}$), compute its singular values $\{\sigma_1, \dots, \sigma_k\}$, normalize them into a probability distribution $p_i = \sigma_i / \sum_j \sigma_j$, and compute the exponential of Shannon entropy:

$$\text{EffectiveRank}\!\left(X^{(\ell)}\right) = \exp\!\left(-\sum_{i=1}^{k} p_i \ln p_i\right)$$

*Note on Sample Scaling:* Because $\text{EffectiveRank}(X^{(\ell)}) \le \min(N, d)$, we verify both small-sample diagnostic runs and scaled evaluation runs ($N = 100$) to ensure spectral measurements are not artificially bottlenecked by sample count.

### 3.3 Linear Probing Accuracy
At each layer $\ell$, we fit an $L_2$-regularized logistic regression classifier on frozen residual activations $x_t^{(\ell)}$ using held-out template splits to prevent lexical memorization. This measures the **linear readability** of contextual sense and structural distinctions at each depth.

### 3.4 Sub-Block Attribution ($\Delta_{\text{Attn}}$ vs. $\Delta_{\text{MLP}}$)
To separate attention-driven routing from nonlinear MLP feature transformation, we track incremental representation shifts before and after each sub-block update within layer $\ell$.

---

## 4. Key Experimental Findings

### 4.1 Early Spectral Expansion Across All Architectures ($N = 100$)
Across all three backbones, Effective Rank rises sharply over the first $4\text{--}6$ transformer blocks before stabilizing on a high-dimensional plateau:
* **`Pythia-410M`:** Expands from $\approx 48$ at Layer 0 to $\approx 76$ by Layer 6, remaining stable around $75\text{--}77$ through Layer 23.
* **`Qwen2.5-1.5B`:** Rises from $\approx 63$ at Layer 0 (with a brief early oscillation across Layers 1–3) into a steady band of $\approx 77\text{--}81$ across middle and late layers.
* **`DeepSeek-R1-Distill-Qwen-1.5B`:** Follows a closely matched spectral envelope ($\approx 67$ at Layer 0 expanding to $\approx 80\text{--}84$), indicating that reasoning distillation preserves global dimensional utilization.

### 4.2 Divergence Between Spectral Breadth and Linear Separability
Comparing Effective Rank against Linear Probing Accuracy on $N = 100$ samples demonstrates that **high dimensionality does not guarantee monotonic linear separability**:
* **`Pythia-410M`:** Exhibits an early probing peak at Layer 2 ($\approx 0.86$), a mid-early dip at Layer 6 ($\approx 0.67$), and a steady recovery to $\approx 0.85$ in late layers.
* **`Qwen2.5-1.5B`:** Dips in early-mid layers ($\approx 0.76$ across Layers 4–8), achieves strong linear separability across middle-to-late layers ($\approx 0.86$ at Layers 15–19), and declines in the final layers ($\approx 0.77\text{--}0.79$ at Layers 24–27) as representations align with next-token output logits.
* **`DeepSeek-R1-Distill-Qwen-1.5B`:** Starts high at Layer 0 ($\approx 0.81$), drops sharply to a trough at Layer 8 ($\approx 0.63$) even while Effective Rank reaches $\approx 81$, and gradually rebuilds to $\approx 0.78$ by Layer 27.

### 4.3 Late-Layer MLP Specialization in Reasoning Distillation
Sub-block attribution reveals a notable final-layer difference between the base and distilled 1.5B models:
* At Layer 27, `DeepSeek-R1-Distill-Qwen-1.5B` exhibits a larger final MLP shift ($\Delta_{\text{MLP}} = 0.2809$) than base `Qwen2.5-1.5B` ($\Delta_{\text{MLP}} = 0.2433$), suggesting heavier late-stage feed-forward refinement immediately prior to unembedding.

---

## 5. Methodological Lessons & Diagnostic Checks

During iterative validation, two critical evaluation hygiene checks were established:
1. **Hook & Shape Verification:** Improper tensor squeezing or architectural module-name mismatches (`gpt_neox.layers` vs. `model.layers`) can produce degenerate rank-1 matrices ($\text{EffectiveRank} = 1.00$). Always assert 2D `(N_samples, d_model)` activation shapes and inspect leading singular values prior to entropy calculation.
2. **Evaluation Granularity:** Small test splits ($N = 8$) introduce $12.5\%$ ($1/8$) discretization artifacts in probing curves and cap effective rank near $\ln(N)$. Scaling evaluation to $N \ge 100$ resolves quantization noise and exposes true layer-wise trajectories.

---

## 6. Limitations & Future Work

* **Observational vs. Causal Claims:** Layer-wise probing and spectral rank characterize representational geometry correlationally; establishing causal necessity requires targeted **activation patching** and head/MLP ablation.
* **Next Planned Experiments:**
  1. **Causal Activation Patching:** Swap $a^{(\ell)}$ and $m^{(\ell)}$ states across contrasting contexts at Layer 8 (`DeepSeek-R1-Distill`) and Layers 24–27 (`Qwen2.5-1.5B`) to measure direct next-token logit flips.
  2. **Nonlinear Probe Comparison:** Train shallow 2-layer MLP probes alongside linear probes at Layer 8 of `DeepSeek-R1-Distill` to test whether sense information is erased or rotated into nonlinear/polysemantic subspaces.
  3. **Further Dataset Scaling:** Expand evaluation templates to $N \ge 500$ with $k$-fold cross-validation across polysemous, syntactic-shift, and monosemous control tiers.

---

## 7. Repository Structure (Recommended)

```text
├── configs/
│   └── models.yaml              # Model IDs, hook layer paths, and tokenizer rules
├── data/
│   └── prompts_level2.json      # Controlled polysemous & syntactic prompt pairs (N=100+)
├── src/
│   ├── extract_activations.py   # Forward hooks for x^(0), attn_out, mlp_out, residual
│   ├── metrics.py               # Centered SVD effective rank, linear probes, delta attribution
│   └── plot_diagnostics.py      # Multi-model layer-wise comparison figures
└── results/
    ├── effective_rank.png
    └── probing_vs_rank_100.png
```
