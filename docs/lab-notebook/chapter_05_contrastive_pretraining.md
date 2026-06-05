# Chapter 5 — Contrastive self-supervised pretraining of a graph encoder for wSMI

> **Goal**: replace the reconstruction-based GAE / VGAE with a MoCo-style
> contrastive encoder that maps each per-epoch wSMI matrix to a 256-D
> embedding (one feature per electrode). The benchmark to beat is the
> raw-wSMI baseline from chapters 1–4: **V = 0.283 ± 0.010** in-sample
> and **pooled LOOCV 3-class bal_acc = 0.59** ([chapter 3
> §3.6e](./chapter_03_optimisations.md#36e-loocv--the-tightest-held-out-estimate)).

> **TL;DR**:
> - Reconstruction wastes capacity modelling within-electrode noise
>   variance; consciousness signal lives in slow, sparse long-range
>   coupling that survives augmentation.
> - **MoCo-v2** wrapper with a **SAGEConv** encoder, **256-D output
>   (1 feature per node)**, projector head only used during pretraining.
> - Positives = augmentations of the **same wSMI matrix** (edge masking,
>   node-feature masking, symmetric Gaussian noise, regional block-drop,
>   small isotropic rescale). Nothing temporal, no electrode permutation.
> - **Key open questions** to settle empirically: temperature τ, queue
>   size K, augmentation strength, latent_dim per node (1 vs 2 vs 4
>   pre-flatten), and whether to use ALL epochs vs `last_100_balanced`
>   for pretraining.

---

## 5.1 Why reconstruction is the wrong loss for wSMI

This is the same argument as in your CoCoEEG-fmri chapter 11, scaled
to graph autoencoders on connectivity matrices:

- **Reconstruction objective rewards modelling input variance.** On
  wSMI, the input variance is dominated by **within-electrode noise
  and short-range coupling**. We saw this directly in chapter 1:
  **PCA(50) on standardised upper-triangle wSMI captures only 11.6 %
  of the total variance**. The remaining 88 % is high-D noise that the
  GAE would faithfully reconstruct.
- **Consciousness signal is the *minority* of variance.** Chapter 4
  showed the discriminative signal lives in long-range posterior
  coupling (P–O at 0.018 vs T–T at 0.001 spread across the 3 clusters).
  These are *small absolute differences* against a noisy connectivity
  background. A reconstruction-trained encoder is structurally biased
  to throw this signal away in favour of preserving high-variance
  noise components.
- **Empirical floor**: our supervised PCA baseline already plateaus at
  V = 0.283 / held-out 0.59. A reconstruction GAE on the same features
  would have to be *worse* than this floor unless the learned features
  happen to ignore most of the reconstruction loss — i.e., unless the
  GAE *fails to reconstruct well* in exactly the dimensions that
  matter.

**Contrastive alternative**: MoCo / SimCLR / BYOL drop the per-sample
reconstruction term. The model is trained to pull augmented views of
the same wSMI matrix close, and push views of different matrices
apart, in a learned embedding space. The crucial property: the
contrastive objective **doesn't require reproducing input variance**.
The model is free to discard high-variance noise that doesn't help
distinguish samples — and to keep slow long-range coupling that's
consistent across augmentations but differs between conscious states.

This is the same logic that made SimCLR / DINO competitive with
supervised pretraining on ImageNet. The novelty here is applying it to
**per-epoch graph-structured connectivity matrices** rather than to
images or timeseries.

---

## 5.2 Objective: MoCo-v2 with InfoNCE

We adopt the MoCo-v2 setup from [He et al. 2020](#references) verbatim:

- **Online encoder** `f_q` with parameters `θ_q` — trained by SGD.
- **Momentum encoder** `f_k` with parameters
  `θ_k ← m·θ_k + (1−m)·θ_q`, no gradient, EMA decay
  **`m = 0.999`** (sweep candidate).
- **Queue** of `K` recent EMA-encoded views, ring-buffered, no gradient.
- **InfoNCE loss** with temperature **`τ = 0.07`** (sweep candidate):

  ℓ(q, k_+, queue) = −log [ exp(q·k_+ / τ) / Σ_{k∈{k_+}∪queue} exp(q·k / τ) ]

- **Projection head** `g` (small MLP, used only during pretraining):
  256 → 128 → 128 with BN + ReLU. Comparison is in the 128-D projected
  space; downstream eval uses the pre-projection 256-D backbone output.

**Queue size considerations**: standard MoCo uses K = 65,536 on ImageNet
(1.2 M images). Scaling to our dataset (~17,500 last_100 epochs):
`K = 4096–8192` is a reasonable default. K = 65k means recycling the
same epoch's views ~4× through the queue — defensible but worth
testing K ∈ {1024, 4096, 16384}.

---

## 5.3 Architecture: SAGEConv encoder → 256-D embedding

### 5.3a Encoder backbone

Reuse the SAGEConv stack from [src/model.py](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/src/model.py)
but stop at the encoder (no decoder) and end at **latent_dim = 1 per
node**:

```
input:  (256 nodes, 256 features per node)   — each node = one electrode,
                                                features = its row of wSMI
graph:  K-NN adjacency on scalp coordinates (k=6, anatomical, fixed)

encoder:
  SAGEConv(256 → 128) + BN + LeakyReLU(0.1) + Dropout(0.2)
  SAGEConv(128 → 64)  + BN + LeakyReLU(0.1) + Dropout(0.2)
  SAGEConv(64  → 32)  + BN + LeakyReLU(0.1) + Dropout(0.2)
  SAGEConv(32  → 16)  + BN + LeakyReLU(0.1) + Dropout(0.2)
  SAGEConv(16  → 1)                                            ← per-node 1-D latent

output: (256 nodes, 1 feature)   →  flatten to (256,)  — the deployed
                                                          embedding
```

**Why per-node 1-D**: the user's spec — "1 feature per node, 256-D
total". The whole-matrix representation lives in a 256-vector whose
geometry corresponds to scalp topology. Downstream cluster /
classification / fingerprinting layers can either use the raw 256-D
or a further-reduced PCA / projection.

### 5.3b Projection head (training only)

```
projector:
  Linear(256 → 128) + BN + ReLU
  Linear(128 → 128)
  L2-normalise
```

Used to compute the InfoNCE loss. **Discarded at inference**: the
256-D backbone output is the deployed embedding. This is the
standard MoCo-v2 trick — the projector lets us learn a more abstract
contrastive head without polluting the downstream representation.

### 5.3c Capacity

| component | params | notes |
|---|---|---|
| Encoder (5 SAGEConv blocks) | ~50 K | tiny — fits easily on any GPU |
| Projector | ~50 K | discarded |
| Momentum encoder | ~50 K | identical clone of encoder |
| Queue (K=8192, 128-D) | 4 MB | RAM only |

Compared to CoCoEEG-fmri's 1.15 M params, this is **20× smaller** —
because the graph structure constrains the inductive bias heavily and
we're aggregating over only 256 nodes, not 4500 patch tokens. We can
afford to start small and scale up if needed.

---

## 5.4 Augmentations: wSMI-specific

The contrastive task is **defined** by what counts as a positive view.
Two augmented views of the *same epoch's* wSMI matrix should land
close in embedding space; views of *different epochs* should land
far apart.

### 5.4a What to keep from the EEG augmentation suite

| augmentation | kept? | wSMI-specific value | reasoning |
|---|---|---|---|
| **NodeFeatureMasking** (= ChannelsDropout) | ✓ | p = 0.10, drop 5–15 % | Some electrodes have higher artefact rates; learning to be robust to "missing" rows is wanted. |
| **GaussianNoise** (symmetric) | ✓ | std = 0.02 (post-normalisation) | wSMI values live in [0, 0.3]; std = 0.02 is small enough to not change the cluster identity. Apply symmetrically: `noise = (E + E^T)/2` so the matrix stays symmetric. |
| **AmplitudeScale** | ✓ | scale ∼ U(0.95, 1.05) | global rescale; preserves all relative coupling patterns. |

### 5.4b What to add (wSMI-specific)

| augmentation | rationale |
|---|---|
| **EdgeMask** | zero a random p = 5–10 % of off-diagonal entries (symmetric mask). Simulates "missing some pairwise measurements"; encourages robust learning that doesn't depend on any single coupling. |
| **RegionBlockDrop** | with p = 0.3, zero an entire region's submatrix (e.g. all T-T entries, or all P-O entries). Extreme augmentation — but the consciousness signal should be redundant enough to survive losing one region pair. Tests this directly. |
| **AsymmetricNodeMask** | mask 1–3 entire rows/columns by setting them to zero. Equivalent to "this electrode failed" — a real EEG scenario. |

### 5.4c What to remove from the EEG / fMRI suite

| augmentation | removed because |
|---|---|
| **TimeMasking** | wSMI is already aggregated over a 0.8 s epoch — no time axis to mask. |
| **FrequencyShift / RandomBandStop / PhaseJitter** | wSMI is already a single-band (theta) measure. No frequency axis to perturb. |
| **ChannelsShuffle** | electrode indices encode anatomical / scalp-topographic structure. Shuffling destroys the spatial prior that the K-NN adjacency exploits. The literature consensus (Sitt 2014, King 2013) is that posterior-vs-frontal *position* carries the signal — randomising electrode order would invalidate the whole approach. |
| **CutOut** | rectangular spatial cutouts don't make sense on a connectivity matrix (no spatial locality in the matrix indexing — it's a relational object). |

### 5.4d Augmentation parameters to sweep

| dial | candidates | what it tests |
|---|---|---|
| GaussianNoise std | 0.005, 0.02, 0.05 | strong noise pushes views apart from true positives → loss collapses |
| EdgeMask p | 0.05, 0.10, 0.20 | too aggressive → views lose the coupling fingerprint |
| RegionBlockDrop p | 0.0, 0.15, 0.3 | tests redundancy in the regional signature |
| compose all vs single-at-a-time | both | are these augmentations multiplicatively complementary? |

---

## 5.5 Pretraining data + downstream evaluation

### 5.5a What to pretrain on

Three candidate partitions:

| partition | n epochs | notes |
|---|---|---|
| `last_100_balanced` (our chapter-3 winner) | 4 200 | best supervised signal density |
| `last_100` (no balancing) | 17 500 | more data, less balanced classes |
| **`all` (full data)** | 132 041 | **proposed default** — contrastive is unsupervised, so class imbalance is irrelevant; more epochs = better negatives |

We propose `all` for pretraining (the model never sees labels — class
imbalance can't bias it), then evaluate the learned encoder on the
same `last_100_balanced` partition that chapters 2–4 use, so
the V / bal_acc numbers are head-to-head comparable.

### 5.5b Downstream evaluation — same harness, swap features

Replace the **PCA(50) features** from our held-out pipeline
([scripts/holdout_prediction.py](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/scripts/holdout_prediction.py))
with **the frozen 256-D encoder embeddings**. Everything else
identical:

- Same `last_100_balanced` partition.
- Same `GroupKFold` by subject (5-fold + LOOCV).
- Same GMM(K=3) on the embeddings.
- Same 3 prediction strategies (modal / hard-FP / soft-FP).
- Same baseline comparison: 0.59 LOOCV 3-class is the floor.

The GAE / contrastive encoder either beats the raw-wSMI baseline
or it doesn't. No moving the goalposts.

### 5.5c What "winning" looks like

Aspirational targets:

| metric | raw-wSMI baseline | GAE / contrastive target |
|---|---|---|
| in-sample Cramér's V | 0.283 | **≥ 0.35** |
| LOOCV pooled 3-class bal_acc | 0.59 | **≥ 0.65** |
| LOOCV pooled 6-class bal_acc | 0.33 | **≥ 0.40** |
| **high_doc recall** (MCS+/MCS−/EMCS) | 48 % | **≥ 60 %** ← the real target |

The MCS recall is the one to chase. Chapter 4's regional analysis
showed cluster 2 (the intermediate / MCS cluster) is flat across
region pairs — the learned representation has the most room to find
new discriminative axes for this group.

---

## 5.6 Implementation plan

### 5.6a New files

| path | what |
|---|---|
| `src/contrastive_model.py` | `GraphMoCo` class wrapping encoder + projector + momentum encoder + queue |
| `src/augmentations.py` | composable wSMI augmentation pipeline |
| `src/train_contrastive.py` | training loop + InfoNCE loss + checkpointing |
| `slurm/contrastive_pretrain.sbatch` | 1×GPU, 12 h, on `parietal` partition |
| `scripts/contrastive_eval.py` | freeze encoder → embed all last_100 → run our existing 5-fold / LOOCV harness with the new features |
| `slurm/contrastive_eval.sbatch` | re-uses the holdout infrastructure |

### 5.6b Files we re-use unchanged

- [src/preprocessing.py](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/src/preprocessing.py) — graph construction, KNN adjacency
- [src/lazy_dataset.py](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/src/lazy_dataset.py) — mmap-backed loader
- [scripts/holdout_prediction.py](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity/scripts/holdout_prediction.py) — only the feature-extraction stage swaps

### 5.6c Sweep / tuning plan

Two waves, each on a single GPU node:

**Wave 1** — sanity check + best-guess defaults. ~6 h. Single run:
- pretraining data: `all`
- encoder: SAGEConv 256→128→64→32→16→1
- augmentations: GaussianNoise(σ=0.02) + EdgeMask(p=0.1) + NodeMask(p=0.1)
- MoCo: K=8192, τ=0.07, m=0.999, batch=128
- 50 pretrain epochs. Then eval against the 0.59 baseline.

**Wave 2** — Ray Tune-style sweep over the dials that wave 1 was most
sensitive to. ~12 h. Likely sweep:
- temperature τ ∈ {0.05, 0.07, 0.15}
- queue K ∈ {4096, 16384}
- augmentation strength (low / medium / high preset)
- latent_dim per node ∈ {1, 2, 4} (does our 256-D constraint help or hurt?)

### 5.6d Honest deliverable timeline

- Implementation: ~1 day to write the three new scripts + smoke test.
- Wave 1: 1 SLURM job, ~6 h on `parietal` GPU.
- Eval: 1 SLURM job, ~1 h (re-uses existing infrastructure).
- Wave 2 sweep: 12 h.

Total wall-clock: 2 days to a clean answer about whether contrastive
beats reconstruction (and the raw-wSMI baseline) on this dataset.

---

## 5.7 Open questions worth flagging up front

1. **Is the K-NN graph the right adjacency?** Our current setup uses
   anatomical scalp K-NN (k=6). Alternative: thresholded
   *functional* adjacency from the wSMI matrix itself
   (top-K edges per node). The functional graph encodes
   coupling-strength priors but changes per epoch — that complicates
   the augmentation (you'd have to perturb the graph and the features
   jointly).
2. **Should we use the GAE encoder weights as warm-start?** We trained
   a GAE briefly in early smoke tests. Using those weights to
   initialise the contrastive encoder might give faster convergence —
   or might overfit to reconstruction structure that the contrastive
   loss is trying to *avoid*. Empirical question.
3. **VGAE-style noise injection?** If we adopt a small KL term that
   regularises the embedding distribution (BYOL / VICReg style), the
   model might generalise better. But this adds a knob to tune; not
   in the minimum-viable pipeline above.
4. **How does LOOCV scale with the new encoder?** Each fold needs
   per-fold pretraining for strict no-leakage. With 144 folds × 6 h
   pretraining, that's prohibitive. Two practical alternatives:
   - **(a)** Pretrain once on subjects we'll never test (impossible —
     we test all subjects).
   - **(b)** Pretrain on ALL data once (mild leakage: the embedding
     captures *unsupervised* structure from test subjects, but we
     never see test labels). This is the standard approach in
     self-supervised neuroimaging (Brain-JEPA, BrainLM, BrainHarmony)
     and we should justify it explicitly.
   - **(c)** Pretrain per-fold for 5-fold (15 hours total).
   We'll start with (b) for the contrastive headline number, and
   report (c) as a leakage-paranoid sensitivity analysis in the
   paper.

---

## 5.8 Status

**This chapter is design / planning only.** No code yet. Once
aligned, we implement in the order:

1. [ ] `src/augmentations.py` — wSMI augmentation pipeline
2. [ ] `src/contrastive_model.py` — encoder + projector + MoCo wrapper
3. [ ] `src/train_contrastive.py` — training loop + checkpointing
4. [ ] `slurm/contrastive_pretrain.sbatch` — submit pretraining
5. [ ] Smoke test on 5–10 subjects to verify loss decreases sanely
6. [ ] Full pretraining run
7. [ ] `scripts/contrastive_eval.py` — embed all epochs, swap into holdout pipeline
8. [ ] Compare against the 0.59 LOOCV baseline
9. [ ] If wins: write up; if loses or ties: tune τ / K / augmentations
       (wave 2 sweep) before concluding the approach doesn't work

---

## References

- Chen X. et al. (2020). *Improved Baselines with Momentum Contrastive
  Learning (MoCo v2)*. arXiv:2003.04297.
- He K. et al. (2020). *Momentum Contrast for Unsupervised Visual
  Representation Learning (MoCo)*. CVPR.
- Caron M. et al. (2021). *Emerging Properties in Self-Supervised
  Vision Transformers (DINO)*. ICCV.
- Grill J.-B. et al. (2020). *Bootstrap Your Own Latent (BYOL)*. NeurIPS.
- Author's own CoCoEEG-fmri chapter 11 (Marraffini, in preparation).
- Sitt J. D. et al. (2014). *Large scale screening of neural signatures
  of consciousness*. **Brain** 137:2258–2270.

---

*See: [methodology.md](./methodology.md) for the supervised baseline
pipeline this design replaces; [chapter 3](./chapter_03_optimisations.md)
and [chapter 4](./chapter_04_regional_interpretation.md) for the
targets to beat.*
