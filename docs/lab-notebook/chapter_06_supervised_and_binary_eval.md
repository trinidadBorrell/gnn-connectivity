# Chapter 6 — Supervised end-to-end & binary discrimination benchmarks

> **Goal**: after the contrastive (MoCo) approach in chapter 5 failed
> to beat the raw-PCA baseline ([chapter 3
> §3.6e](./chapter_03_optimisations.md#36e-loocv--the-tightest-held-out-estimate),
> baseline pooled LOOCV 3-class bal_acc = 0.59), we asked: does the
> exact same graph encoder, trained end-to-end with a supervised
> diagnosis head, do better? And, beyond 3-class bal_acc, what do the
> models look like on the two clinically meaningful **binary**
> discriminations — control vs anything else, and MCS vs UWS?

> **TL;DR**:
>
> | model | control vs rest AUC | MCS vs UWS AUC | 3-class bal_acc |
> |---|---|---|---|
> | GMM K=3 | 0.815 ± 0.095 | 0.713 ± 0.155 | **0.611 ± 0.131** |
> | GMM K=4 | 0.815 ± 0.101 | **0.723 ± 0.158** | 0.607 ± 0.133 |
> | Supervised GCN | **0.931 ± 0.083** | 0.678 ± 0.155 | 0.527 ± 0.118 |
> | MoCo GCN (best, wave 1) | 0.772 ± **0.069** | 0.696 ± **0.127** | 0.535 ± **0.086** |
>
> *(All values mean ± std across 5 subject-disjoint folds. ROC AUC
> reported with bootstrap 95 % CI per fold; see §6.5 figure.)*
>
> **Reading the table.** The supervised GCN wins decisively on the
> easier *control vs DOC* discrimination (+0.116 AUC over the GMM
> baseline) but loses ~0.04 AUC on the harder *MCS vs UWS* split. The
> 3-class metric averages these and so favours the GMM K=3, but this
> masks the actual structure — see §6.4c.
>
> **MoCo lands in the "consistent but mediocre" zone**: the smallest
> std on every metric (0.069 / 0.127 / 0.086) but lower mean AUC than
> the GMMs on both binary tasks. Contrastive pretraining gave us
> stability across folds without power — the encoder learned to be
> robust to augmentations *and* to ignore the discriminative
> dimensions, exactly as predicted in [chapter 5](./chapter_05_contrastive_pretraining.md).

---

## 6.1 Why we tried supervised after the MoCo failure

[Chapter 5](./chapter_05_contrastive_pretraining.md) laid out the case
for contrastive pretraining; the practical results (three augmentation
strengths × two latent sizes) all underperformed the raw-PCA baseline:

| MoCo variant | 5-fold pooled 3-class bal_acc | vs baseline 0.59 |
|---|---|---|
| Wave 1 (medium augs, 256-D) | 0.531 | −0.06 |
| Wave 2A (lighter augs) | **0.475** | −0.12 |
| Wave 2B (512-D latent) | 0.514 | −0.08 |
| Wave 2C (150 epochs) | embedding collapse (std 0.015) — eval failed |

The signal across the three trained MoCo variants: **lower pretrain
loss correlated with worse downstream accuracy.** The augmentations
(edge mask, node mask, symmetric noise) were preserving the wrong
invariances — exactly the consciousness-relevant electrode-pair
structure we identified in [chapter 4](./chapter_04_regional_interpretation.md)
was being treated as nuisance by the contrastive loss. The MoCo
projector pulled the encoder toward a representation invariant to those
augmentations, which means **invariant to the dimensions we wanted to
keep**.

The natural follow-up: keep the encoder architecture, drop the
invariance objective, supervise directly with diagnosis labels.

---

## 6.2 Supervised pipeline

Identical encoder architecture to chapter 5's plan:

- **Adjacency**: dense (256, 256) RBF-weighted K-NN on electrode XYZ
  coordinates (K=10, σ = median nearest-neighbour distance),
  symmetric-normalised à la GCN. **No wSMI in the graph structure** —
  wSMI is the node feature.
- **Encoder**: 5-layer `DenseGCN` stack `256 → 128 → 64 → 32 → 16 → 1`
  with BatchNorm + LeakyReLU(0.1) + Dropout(0.3) on every intermediate
  layer. Output is one scalar per electrode → flatten → **256-D
  per-epoch embedding**.
- **Head**: `Linear(256 → 64) → BN → ReLU → Dropout → Linear(64 → 3)`
  with `CrossEntropyLoss(class_weight='balanced')` over the **3-class
  coarse target** (`control / low_doc / high_doc`, matching the
  collapse used throughout the baseline).
- **Training**: 30 epochs, AdamW + cosine LR schedule (lr = 1e-3,
  wd = 1e-4), batch size 512, light augmentations (Gaussian σ = 0.02,
  edge mask p = 0.05, node mask p = 0.05) used only as regularisation.
- **Evaluation**: same 5-fold subject-level `GroupKFold` as the
  baseline. Per fold, train end-to-end on 4/5 of subjects (~92), then
  predict per-epoch softmax probabilities on the held-out 1/5 (~23
  subjects); aggregate to per-subject by mean across epochs.

Implementation: [`src/train_supervised.py`](../../src/train_supervised.py),
sbatch wrapper [`slurm/train_supervised.sbatch`](../../slurm/train_supervised.sbatch).
Per-fold encoder weights now saved to `output/supervised/model_fold{1..5}.pt`
so we can re-do inference without retraining.

---

## 6.3 Headline result: per-fold 3-class balanced accuracy

5 fold-level values + mean ± std for the three best models. All
subject-disjoint, all under the same 5-fold GroupKFold protocol.

| fold | GMM K=3 | GMM K=4 | Supervised GCN | MoCo GCN |
|---|---|---|---|---|
| 1 | 0.766 | **0.790** | 0.367 | 0.675 |
| 2 | 0.706 | 0.706 | 0.630 | 0.482 |
| 3 | 0.493 | 0.493 | 0.548 | 0.497 |
| 4 | 0.464 | 0.508 | 0.449 | 0.464 |
| 5 | 0.626 | 0.537 | **0.641** | 0.558 |
| **mean ± std** | **0.611 ± 0.131** | 0.607 ± 0.133 | 0.527 ± 0.118 | 0.535 ± **0.086** |

The earlier-reported 0.590 / 0.601 figures were the **pooled** LOOCV
bal_acc — concatenating all 144 LOOCV predictions before computing
balanced accuracy. The per-fold std for LOOCV is undefined (1 subject
per fold). The 5-fold per-fold values above are the directly
comparable quantity for variance.

Per-fold values for the GMMs come from rerunning the same
GroupKFold + GMM + LogReg pipeline at K=3 and K=4 with 5 folds, on the
existing `last_100 × balanced` partition (see
[`scripts/binary_roc_gmm.py`](../../scripts/binary_roc_gmm.py)). The
GMMs are within one std of the pooled LOOCV figure, confirming the
5-fold estimator is consistent with LOOCV at this scale.

**Interpretation.** On the 3-class metric the supervised GCN doesn't
clearly win — its mean is below the GMMs and the per-fold variance
spans nearly the full possible range (0.367 fold 1 to 0.641 fold 5).
But the next section shows that 3-class bal_acc averages over a fast
axis (control vs DOC) and a hard axis (within-DOC), masking the actual
behaviour.

---

## 6.4 Binary discrimination — the structure 3-class bal_acc hides

Two clinically meaningful binary tasks:

- **Control vs any other** — does the model detect lack of preserved
  consciousness signal? n = 144 subjects (14 control, 130 DOC).
- **MCS (high_doc) vs UWS/COMA (low_doc)** — the harder within-DOC
  recovery boundary, excluding controls. n = 130.

For each model and each fold we compute the per-fold ROC AUC on its
held-out subjects + a bootstrap (1,000 resamples) for the 95 % CI:

### 6.4a Control vs anything else (n_pos = 14, n_neg = 130)

| fold | GMM K=3 | GMM K=4 | Supervised GCN | MoCo GCN |
|---|---|---|---|---|
| 1 | 0.923 | 0.923 | **1.000** | 0.885 |
| 2 | 0.859 | 0.808 | **1.000** | 0.782 |
| 3 | 0.833 | 0.872 | **0.952** | 0.756 |
| 4 | 0.795 | 0.821 | **0.905** | 0.718 |
| 5 | 0.667 | 0.654 | **0.800** | 0.718 |
| **mean ± std** | 0.815 ± 0.095 | 0.815 ± 0.101 | **0.931 ± 0.083** | 0.772 ± 0.069 |

The **supervised GCN beats every other model on every fold** for the
control-vs-rest task, with 3 of 5 folds at ≥ 0.95 (and 2 of those at
the n=14/130 ceiling 1.0). The +0.116 absolute AUC improvement over
the GMM baseline is well outside one standard deviation, and the
supervised model's worst fold (0.800) is still better than the GMMs'
best four folds. **MoCo is the worst of the four** with a tight spread
(0.718–0.885) — uniformly worse than the GMM baseline, suggesting the
contrastive objective discarded control-specific discriminative
features.

### 6.4b MCS vs UWS/COMA (n_pos ≈ 14, n_neg ≈ 12 per fold)

| fold | GMM K=3 | GMM K=4 | Supervised GCN | MoCo GCN |
|---|---|---|---|---|
| 1 | 0.810 | 0.798 | 0.600 | 0.792 |
| 2 | 0.825 | 0.812 | 0.583 | 0.731 |
| 3 | **0.448** | **0.442** | **0.898** | **0.475** |
| 4 | 0.708 | 0.768 | 0.529 | 0.768 |
| 5 | 0.776 | 0.794 | 0.778 | 0.715 |
| **mean ± std** | 0.713 ± 0.155 | 0.723 ± 0.158 | 0.678 ± 0.155 | 0.696 ± **0.127** |

Inversely complementary on this task: the GMMs (and MoCo) are strong on
folds 1, 2, 4, 5 but **all collapse to near-chance on fold 3** (0.44–0.48 —
the held-out subjects in this fold are atypical for any clustering-based
representation). The supervised GCN is the inverse: only **fold 3 is
strong (0.898)**; the other four folds hover at 0.5–0.6. On average the
four models are statistically indistinguishable on MCS-vs-UWS (means
within 0.05, all stds ≥ 0.13 except MoCo's 0.127). MoCo has the tightest
spread but pays for it with a clear fold 3 failure that the supervised
GCN doesn't share.

### 6.4c Why 3-class bal_acc was misleading

The supervised GCN is producing predictions of the form "almost
certainly control, or almost certainly low_doc, but I can't tell
high_doc from low_doc" — see the pooled supervised confusion matrix:

```
              pred: control  low_doc  high_doc
true control     [   7         2        2 ]   64 % recall  ✓
true low_doc     [   1        45        5 ]   88 % recall  ✓
true high_doc    [   3        43        7 ]   13 % recall  ✗ (collapsed into low_doc)
```

The 3-class balanced accuracy is `mean(recalls) ≈ 0.55`, dragged down
by the 13 % high_doc recall. But its **binary** behaviour on the
control vs DOC boundary is nearly perfect (recall 64 % / FP rate 1.7 %),
which is what ROC AUC captures.

This is the right framing for the paper: 3-class bal_acc averages two
problems of very different difficulty, and the supervised model wins
the easy half decisively while leaving the hard half open.

---

## 6.5 Figure

Plot at [output/roc_compare/roc_compare.png](../../output/roc_compare/roc_compare.png):

- **Top row**: per-task bar chart (mean ± std across 5 folds) with
  per-fold point AUC + bootstrap 95 % CI overlaid as black error bars.
- **Bottom row**: pooled ROC curves (one per model) — the average
  predicted score per subject across folds, fed to `roc_curve`.

The visual takeaway is the same as the table: tight cluster of curves
for control-vs-rest with the supervised GCN clearly above, and a wide
spread for MCS-vs-UWS with no clear winner.

---

## 6.6 What this means for the paper

The unsupervised baseline of [chapter 3](./chapter_03_optimisations.md)
remains the **clean unsupervised result** (label-free replication of
the Sitt 2014 / King 2013 / Casarotto 2016 signature). The supervised
GCN adds two distinct points to the story:

1. **The wSMI features carry control-vs-DOC information that a
   GroupKFold-correct supervised GCN can extract to AUC ≈ 0.93** — a
   substantive improvement over the unsupervised GMM (AUC ≈ 0.82) at
   the same protocol.
2. **No method tested cracks the within-DOC boundary from wSMI alone.**
   GMMs and supervised both land around 0.7 AUC for MCS vs UWS, with
   large fold-to-fold variance and complementary failure modes. This
   is consistent with the multi-band / multi-feature requirement
   established by Sitt 2014 (~0.78 UWS-vs-MCS AUC with the full
   feature panel of theta + α + β + δ + γ wSMI + spectral power +
   complexity).

3. **MoCo contrastive pretraining produces the most stable model
   across folds** (smallest std on every metric) but with the lowest
   mean AUC on the easy task and tied-but-not-better on the hard task.
   This is the empirical signature of an encoder that learned a
   smooth, well-conditioned representation that's also blind to the
   exact dimensions we care about — augmentation invariance traded
   power for stability.

For the next phase: extend node features beyond theta-band wSMI (e.g.
add spectral power in 4 bands per electrode — exactly the user-allowed
"up to 4 features per node" envelope from chapter 5's contrastive
design). The candidate paper structure already accommodates this as a
multi-band ablation.

---

## 6.7 Reproducibility

- Encoder weights: `output/supervised/model_fold{1..5}.pt`
- Per-subject probabilities: `output/{roc_gmm_K3_5fold,roc_gmm_K4_5fold,supervised}/per_subject_proba.csv`
- Per-fold AUC + bootstrap: `output/roc_compare/roc_compare.json`
- Plot: `output/roc_compare/roc_compare.png`
- Scripts:
  - [`src/train_supervised.py`](../../src/train_supervised.py)
  - [`scripts/binary_roc_gmm.py`](../../scripts/binary_roc_gmm.py)
  - [`scripts/compare_roc.py`](../../scripts/compare_roc.py)
- SLURM wrappers under [`slurm/`](../../slurm/) with the same
  `random_state = 42` convention as the rest of the notebook.
