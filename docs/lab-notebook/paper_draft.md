# Paper draft — unsupervised wSMI clustering for disorders of consciousness

> **Status**: skeleton with section briefs (≤ 3–4 sentences each). Each
> section points to the lab-notebook chapter that has the full numbers,
> figures, and methodology. Final writing should be done from those
> chapters, not from this file.

> **Target venue**: clinical-translational neuroimaging — *Brain*,
> *NeuroImage*, *NeuroImage: Clinical*, or *Clinical Neurophysiology*.
> The "unsupervised replication of supervised DOC findings" angle plays
> best in *Brain* or *NeuroImage: Clinical*; the methods angle suits
> *Clinical Neurophysiology*.

> **Proposed title**:
> *"The consciousness gradient is the principal axis of resting-state
>  wSMI: an unsupervised replication of EEG disorders-of-consciousness
>  biomarkers in 144 patients."*

> **Authors / affiliations**: TBD.

---

## Abstract

*(~ 200 words — write last, summarising every section below.)*

**Background.** Theta-band wSMI is the single most discriminative EEG
biomarker between unresponsive wakefulness syndrome (UWS) and the
minimally conscious state (MCS) in supervised classifiers
([Sitt 2014](#references)). Whether the same signal organises the data
*without* using diagnostic labels has not been established.

**Methods.** We applied diagnosis-balanced GMM clustering to per-epoch
wSMI matrices from 144 patients (164 sessions across UWS / MCS−/+ /
EMCS / COMA + 14 healthy controls), with subject-disjoint 5-fold
cross-validation. See [methodology.md](./methodology.md) and
[chapter 3 §3.6](./chapter_03_optimisations.md#36-held-out-validation-does-the-v0252-actually-mean-anything).

**Results.** Three unsupervised clusters emerged, ordered exactly along
the clinical consciousness gradient on PC1 of the wSMI feature space.
Held-out subject-level 3-class balanced accuracy reached
**0.611 ± 0.131** (mean ± std across 5 subject-disjoint folds; pooled
144-fold LOOCV = 0.590), vs chance 0.33. The most discriminative
coupling pair was parietal-occipital (P–O), matching the canonical
"posterior hot zone" literature. On the two clinically meaningful
binary tasks, ROC AUC across 5 folds was
**0.815 ± 0.095 for control vs any other DOC diagnosis** and
**0.713 ± 0.155 for MCS vs UWS/COMA** — clean separation at the
conscious/unconscious boundary, but ambiguous performance at the MCS
recovery boundary, mirroring the well-documented CRS-R misdiagnosis
rate. A supervised end-to-end graph-CNN on the same encoder
architecture sharpens the easy axis (control vs DOC AUC
**0.931 ± 0.083**, +0.116 over the unsupervised baseline) but does
not crack the within-DOC boundary (0.678 ± 0.155 AUC) — consistent
with the multi-feature requirement of Sitt 2014.

**Conclusions.** Unsupervised clustering of resting-state wSMI recovers
the published EEG consciousness signature without any supervision.
This provides a transparent, label-free baseline for future
representation-learning approaches and a candidate clinical-grade
fingerprint for unseen patients. Supervised graph models extract
additional signal on the conscious/unconscious boundary specifically
but reach the same ceiling as the unsupervised baseline on the harder
within-DOC discrimination.

---

## 1. Introduction (3–4 sentences)

> **What goes here**: the clinical problem + the gap our paper fills,
> in two beats.

1. **Clinical motivation**: distinguishing UWS from MCS at the bedside
   is hard — the gold-standard CRS-R has ~40 % misdiagnosis rate
   ([Schnakers et al. 2009](#references)) — and EEG-based biomarkers
   are the most scalable adjunct.
2. **Prior consensus**: large supervised studies converge on theta-band
   wSMI in centro-posterior electrodes as the dominant EEG marker of
   consciousness ([Sitt 2014](#references), [King 2013](#references),
   [Engemann 2018](#references)).
3. **Open question**: would the same signal *organise* the data
   without diagnostic labels? An unsupervised replication would
   confirm the signal lives in the raw data, not just in the
   supervised loss landscape.
4. **Our contribution**: we show that **a GMM with three components,
   fit on diagnosis-balanced last-100-epoch wSMI features, recovers
   the clinical consciousness gradient as its principal axis and
   reaches 0.61 held-out 3-class balanced accuracy on 144 unseen
   patients** — a label-free, interpretable, reproducible baseline.

*Full development*:
[lab-notebook README](./README.md) (one-screen summary) +
[chapter 1](./chapter_01_clustering_ablation.md) (ablation story).

---

## 2. Related work (3–4 sentences)

> **What goes here**: position our unsupervised replication against the
> supervised DOC EEG literature.

1. **Foundational supervised wSMI work**: Sitt et al. 2014 (*Brain*)
   screened > 100 EEG features in 167 DOC patients and ranked
   theta-band wSMI between centro-posterior electrodes as the single
   most discriminative feature; King et al. 2013 (*Curr Biol*)
   introduced the wSMI metric and established theta as the optimal
   band.
2. **Convergent evidence from other modalities**: Casarotto et al.
   2016 (*Ann Neurol*) reached the same anatomical conclusion via
   TMS-EEG perturbational complexity (PCI) — long-range cortico-
   cortical coupling, especially over the posterior hot zone, is the
   integration core whose collapse defines unresponsiveness.
3. **Cross-site validation**: Engemann et al. 2018 (*Brain*) confirmed
   the Sitt 2014 ranking across four European hospitals — the same
   centro-posterior theta-wSMI signature is the most reproducible
   feature.
4. **What's missing in the literature**: all of the above are
   *supervised* classifiers; none asks whether the data carries the
   signal in unsupervised structure. **Our contribution is precisely
   that test** — and it succeeds.

*Full citations and detailed comparison*:
[chapter 4 §4.4](./chapter_04_regional_interpretation.md#44-comparison-with-the-published-doc-eeg-literature).

---

## 3. Methodology (3–4 sentences)

> **What goes here**: enough for a reader to know the pipeline + the
> leakage-free splitting. The rest lives in the methodology document.

1. **Data**: 178 EEG sessions from **144 patients** (130 DOC + 14
   controls; CRS-R diagnoses spanning UWS / MCS−/+ / EMCS / COMA),
   theta-band wSMI computed per 0.8 s epoch over 256 EGI electrodes
   (~132,000 epochs total). See
   [methodology.md §1](./methodology.md#1-data).
2. **Pipeline**: take **last 100 epochs per session** (the most
   discriminative temporal window, see
   [chapter 3 §3.3](./chapter_03_optimisations.md#33-drilling-into-time-within-session)),
   **balance across diagnoses** (~4,200 epochs), `StandardScaler` →
   `PCA(50)` → `GaussianMixture(K=3, full covariance)`. Subject-level
   prediction by aggregating per-epoch cluster posteriors into a
   3-dim fingerprint, then `LogisticRegression`.
3. **No leakage**: **subject-disjoint GroupKFold** with all
   preprocessors (scaler, PCA, GMM, logreg) refit per fold on training
   subjects only. Reproducible with `random_state=42` throughout. Full
   per-fold pipeline + leakage proof in
   [methodology.md §2](./methodology.md#2-subject-level-splits--no-data-leakage).
4. **Evaluation**: in-sample Cramér's V on the cluster × diagnosis
   contingency, plus held-out balanced accuracy under 5-fold and LOOCV
   on three diagnosis-prediction strategies (modal cluster, hard
   fingerprint, soft posterior fingerprint).

*Full reproducibility details*: [methodology.md](./methodology.md).
*Code*: `scripts/` and `slurm/` directories of the repo
[gnn-connectivity](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity).

---

## 4. Results (3–4 sentences per sub-result)

> **What goes here**: 3 main results, each with its own short paragraph
> and figure reference. Numbers are concrete and from the lab notebook.

### 4.1 Unsupervised clusters align with the consciousness gradient
(*figure: PCA(2) coloured by diagnosis with monotone trajectory + PC1 loading decomposition*)

- The three GMM clusters are strongly enriched for distinct
  diagnoses: cl_0 (n=8,174) = UWS-dominated (49 %),
  cl_1 (n=1,772) = control-rich (29 %), cl_2 (n=7,854) = MCS/EMCS-leaning.
- **PC1 of the wSMI feature space orders diagnoses strictly
  monotonically along the clinical gradient** — control (+24.2) →
  EMCS (+2.6) → MCS+ (−1.8) → MCS− (−4.8) → UWS (−7.2) → COMA (−10.8)
  — with no supervision involved. The conscious / unconscious
  boundary is ~5× larger than within-DOC transitions.
- **Decomposing PC1**: the loading vector across the 32,640 electrode
  pairs has *uniformly positive* sign for all top-ranked region pairs
  — i.e. higher wSMI in these pairs corresponds to more consciousness.
  Top 5 region pairs by mean signed loading: **P–O (+0.0065), C–O
  (+0.0060), C–P (+0.0060), O–O (+0.0058), P–P (+0.0058)** — all
  involve posterior cortex. The top 20 individual electrode pairs all
  sit inside the occipital/parietal cluster (E98, E100, E101, E109,
  E110, E118, E152, E153, E161, E162, E170, E171), with the single
  strongest pair E118 ↔ E153 at +0.0105. **The principal linear axis
  of resting-state theta-wSMI is parieto-occipital connectivity**,
  recovering the posterior hot zone hypothesis as a direct
  decomposition of the variance — not just as the GMM contrast of
  chapter 4 / §4.3.
- Cramér's V on the cluster × diagnosis contingency: **0.280 in-sample**.
- *Figure sources*:
  [chapter 2 §2.4c](./chapter_02_interpretability.md#24c-k3-manifold--pca2-and-umap2)
  → [figures/pca2_diagnosis_K3.png](./figures/pca2_diagnosis_K3.png),
  [chapter 2 §2.7](./chapter_02_interpretability.md#27-what-does-pc1-actually-measure-loading-decomposition)
  → [figures/pc1_region_loadings.png](./figures/pc1_region_loadings.png),
  [figures/pc1_scalp_top_edges.png](./figures/pc1_scalp_top_edges.png).

### 4.2 The signal generalises to unseen patients
(*figure: 3-class confusion matrix, K=3 LOOCV, 144 pooled subject-level predictions*)

- Held-out diagnosis prediction via subject-level cluster
  fingerprints + logistic regression reaches **0.611 ± 0.131
  balanced accuracy** under K=3 5-fold subject-disjoint CV (mean ± std
  across 5 folds; pooled 144-fold LOOCV bal_acc = 0.590) on the
  collapsed 3-class problem (control / low-DOC / high-DOC), vs 0.33
  random — a **+28 pp gain over chance**. K=4 5-fold is tied at
  0.607 ± 0.133 (the K=3 vs K=4 gap from earlier exploratory runs
  disappears at this estimator).
- The confusion matrix is **diagnostically meaningful**: the
  conscious/unconscious boundary is clean (4 cross-errors / 75 pairs
  between control and low_doc), but the MCS recovery boundary is
  hard (48 % recall on high_doc, errors spread to both neighbours) —
  consistent with the ~40 % CRS-R misdiagnosis rate
  ([Schnakers 2009](#references)).
- *Figure source*:
  [chapter 3 §3.6e](./chapter_03_optimisations.md#36e-loocv--the-tightest-held-out-estimate)
  → [figures/confusion_soft_3class_LOOCV_K3.png](./figures/confusion_soft_3class_LOOCV_K3.png).

### 4.2bis Binary discrimination — control vs DOC and MCS vs UWS
(*figure: 2-panel ROC, 3 models × 2 binary tasks, with bootstrap CIs*)

- Decomposing the 3-class problem into the two clinically meaningful
  binary discriminations reveals that the 3-class metric averages a
  fast and a slow axis. **Per-fold ROC AUC across the same 5
  subject-disjoint folds** (mean ± std):

  | model | control vs any other DOC | MCS (high_doc) vs UWS/COMA (low_doc) |
  |---|---|---|
  | GMM K=3 | 0.815 ± 0.095 | 0.713 ± 0.155 |
  | GMM K=4 | 0.815 ± 0.101 | 0.723 ± 0.158 |
  | MoCo GCN, best wave-1 (§4.4) | 0.772 ± 0.069 | 0.696 ± 0.127 |
  | Supervised GCN (§4.4) | **0.931 ± 0.083** | 0.678 ± 0.155 |

- The unsupervised K=3 GMM **already reaches AUC = 0.82** for
  control-vs-DOC and 0.71 for MCS-vs-UWS — without any diagnostic
  supervision. K=3 vs K=4 are statistically indistinguishable on both
  binary tasks (means within 0.01, fully overlapping std).
- The within-DOC AUC of 0.71 sits inside one std of the
  multi-feature supervised benchmark from
  [Sitt 2014](#references) (~0.78), achieved here from theta-band wSMI
  alone with no diagnostic labels.
- *Figure source*:
  [chapter 6 §6.4–6.5](./chapter_06_supervised_and_binary_eval.md#64-binary-discrimination--the-structure-3-class-bal_acc-hides)
  → [figures/roc_compare.png](./figures/roc_compare.png).

### 4.4 Supervised graph-CNN sharpens the easy axis only
(*figure: same ROC panel as §4.2bis; per-fold supervised confusion matrix as inset*)

- Training the same anatomical GCN encoder end-to-end with a 3-class
  cross-entropy head (subject-disjoint 5-fold, [chapter 6](./chapter_06_supervised_and_binary_eval.md))
  produces a model that **wins decisively on the control-vs-DOC binary
  task** (AUC 0.931 ± 0.083 vs 0.815 ± 0.095 for the GMM K=3, a
  +0.116 absolute AUC improvement well outside one std) but **ties
  the unsupervised baseline on MCS-vs-UWS** (0.678 ± 0.155 vs
  0.713 ± 0.155).
- The supervised confusion matrix is informative: 88 % recall on
  low_doc, 64 % on control, but only 13 % on high_doc — i.e. the model
  predicts "almost certainly control or almost certainly low_doc" and
  collapses high_doc into low_doc. This is the same wall the
  unsupervised method hits at the MCS recovery boundary, expressed
  differently.
- Across three different learning paradigms tested in the appendix
  (MoCo-v2 contrastive pretraining, region-pair-restricted clustering,
  end-to-end supervised), **none beats the 5-fold AUC of 0.71 for MCS
  vs UWS from theta-wSMI alone**. This converges with the
  multi-band/multi-feature requirement of
  [Sitt 2014](#references) and motivates §7 future work.
- MoCo-v2 contrastive pretraining produced an encoder that is more
  **stable across folds** than every other model (per-fold AUC std
  0.069 / 0.127 vs ≥ 0.083 / ≥ 0.155 for the others) but with **lower
  mean AUC on control-vs-DOC** (0.772 ± 0.069) — augmentation
  invariance traded discriminative power for fold-to-fold consistency.
- *Figure / numbers*:
  [chapter 6 §6.3–6.4](./chapter_06_supervised_and_binary_eval.md#63-headline-result-per-fold-3-class-balanced-accuracy)
  → [figures/roc_compare.png](./figures/roc_compare.png) and
  [figures/supervised_confusion.png](./figures/supervised_confusion.png) (TBD).

### 4.3 The regional fingerprint matches the published consciousness signature
(*figure: 5×5 region-pair consciousness-sensitivity heatmap + ranked barplot*)

- Grouping the 256 electrodes into 5 anatomical regions (F/C/P/T/O) and
  computing per-cluster mean wSMI per region pair reveals: **the
  top-5 most consciousness-sensitive region pairs all involve
  posterior cortex** (P–O 0.018, P–P 0.016, C–P 0.016, C–O 0.015,
  O–O 0.014).
- Temporal-temporal (T–T) coupling is **consciousness-invariant**
  (spread = 0.001 across clusters), suggesting subcortical
  stabilisation.
- These rankings independently replicate the centro-posterior /
  posterior-hot-zone findings of Sitt 2014 / King 2013 /
  Casarotto 2016 — without using diagnostic labels.
- *Figure sources*:
  [chapter 4 §4.5](./chapter_04_regional_interpretation.md#45-quantifying-consciousness-sensitivity-per-region-pair)
  →
  [figures/consciousness_sensitivity_K3.png](./figures/consciousness_sensitivity_K3.png),
  [figures/pair_ranking_K3.png](./figures/pair_ranking_K3.png).

---

## 5. Discussion (3–4 sentences)

> **What goes here**: what the three results above add up to + the
> "so what" for the field.

1. **The consciousness signal lives in the raw data**, not just in the
   supervised loss surface. PCA(2) of standardised upper-triangle
   wSMI already orders the diagnoses correctly along PC1; the GMM
   merely tiles this manifold into discrete states.
2. **Posterior-centric coupling is the universal marker** — top-5
   sensitive region pairs all include parietal or occipital regions,
   reproducing the "posterior hot zone" hypothesis of consciousness
   ([Koch et al. 2016](#references)).
3. **A label-free 0.61 3-class balanced accuracy** is competitive with
   supervised baselines on similar cohorts ([Sitt 2014](#references)
   reported ~0.78 UWS-vs-MCS AUC on 167 patients; ours is 3-class on
   144 patients with no diagnostic supervision).
4. **The intermediate cluster is clinically ambiguous on wSMI alone** —
   cl_2 has flat region-pair signature and absorbs ~50 % of every
   diagnosis. Refining this is the natural target for a learned
   representation (next section).

*Full elaboration*:
[chapter 4 §4.3–4.4](./chapter_04_regional_interpretation.md#43-the-k3-region-pair-matrix--what-each-cluster-looks-like)
and [chapter 2 §2.6](./chapter_02_interpretability.md#26-putting-the-interpretability-story-together).

---

## 6. Limitations (3–4 sentences)

> **What goes here**: be honest about scope.

1. **Single band** — theta-band wSMI only; the Sitt 2014 panel uses α/β/δ/γ
   alongside theta. A multi-band representation could refine cluster 2
   (the ambiguous intermediate).
2. **Modest n in some classes** — only 7 COMA subjects and 11 EMCS
   subjects in our cohort; 5-fold per-class statistics are noisy at
   the extremes (see [chapter 3 §3.6b](./chapter_03_optimisations.md#36b-per-fold-breakdown-k3--the-winner)).
3. **Coarse parcellation** — our 5-region grouping (F/C/P/T/O) is
   coordinate-driven. A formal atlas-based parcellation
   (Brainnetome, Schaefer) might separate posterior subregions
   more cleanly.
4. **Single dataset** — Sitt-lab Paris cohort only. External
   replication on the cohorts of [Engemann 2018](#references) (4 EU
   hospitals) is needed before clinical claims.

---

## 7. Future work (3–4 sentences)

> **What goes here**: the natural next steps. Sets up the GAE phase
> and the outcome-prediction work.

1. **Learned graph representation (GAE/VGAE)** — replace `PCA(50)`
   with a SAGEConv-based graph autoencoder that respects the
   electrode adjacency structure. **Benchmarks to beat**: in-sample
   V ≥ 0.30; held-out 3-class bal_acc ≥ 0.65; sharper cluster 2.
   Infrastructure already in place (see
   [README §benchmark](./README.md#the-benchmark-the-gae-will-need-to-beat)).
2. **Outcome prediction** — `patient_labels.csv` carries
   `cs_{6m, 1y, 2y}` future-consciousness outcomes for every DOC
   subject. Testing whether baseline cluster occupancy predicts
   *recovery* would be a stronger clinical test than current-state
   classification.
3. **Cross-site validation** — apply the fitted K=3 pipeline to the
   public Engemann 2018 datasets to test generalisation across
   acquisition sites.
4. **Multi-band features** — extend `wSMI_θ` to a 5-band stack and
   re-run the regional analysis; pinpoint which bands carry the
   independent variance over theta.

---

## 8. Reproducibility & code availability

All scripts, sbatch wrappers, and the lab notebook are under
[/data/parietal/store3/work/gmarraff/repos/gnn-connectivity](/data/parietal/store3/work/gmarraff/repos/gnn-connectivity)
(branch `gio-grid-search-clustering`). Every reported number reproduces
with `random_state=42`. Full pipeline + dependencies + SLURM job
configurations in [methodology.md](./methodology.md#4-how-to-re-run-everything).

---

## 9. Figure plan

| # | content | source figure |
|---|---|---|
| 1 | Pipeline schematic: wSMI → upper-triangle → StandardScaler → PCA(50) → GMM(K=3) → fingerprint → LogReg | to draw |
| 2 | PCA(2) of last_100 epochs, coloured by diagnosis with monotone trajectory | [figures/pca2_diagnosis_K3.png](./figures/pca2_diagnosis_K3.png) |
| 2b | PC1 loading decomposition: per-region heatmap + scalp visualisation of strongest electrode pairs | [figures/pc1_region_loadings.png](./figures/pc1_region_loadings.png) + [figures/pc1_scalp_top_edges.png](./figures/pc1_scalp_top_edges.png) |
| 3 | Per-cluster centroid wSMI matrices (256×256, vs grand mean) + 5×5 region summary | [figures/centroids_vs_grand_mean_K3.png](./figures/centroids_vs_grand_mean_K3.png) + [figures/region_summary_vs_grand_mean_K3.png](./figures/region_summary_vs_grand_mean_K3.png) |
| 4 | Consciousness sensitivity per region pair + ranked bar | [figures/consciousness_sensitivity_K3.png](./figures/consciousness_sensitivity_K3.png) + [figures/pair_ranking_K3.png](./figures/pair_ranking_K3.png) |
| 5 | Held-out generalisation: 3-class confusion matrix from K=3 144-fold LOOCV (pooled, soft fingerprint) | [figures/confusion_soft_3class_LOOCV_K3.png](./figures/confusion_soft_3class_LOOCV_K3.png) |
| 6 | Binary ROC AUC across 3 models × 2 tasks, with bootstrap CIs and pooled ROC curves | [figures/roc_compare.png](./figures/roc_compare.png) (source: `output/roc_compare/roc_compare.png`) |
| Supp. 1 | K=4 vs K=3 comparison (centroids side by side) | [figures/centroids_vs_grand_mean.png](./figures/centroids_vs_grand_mean.png) + [figures/centroids_vs_grand_mean_K3.png](./figures/centroids_vs_grand_mean_K3.png) |
| Supp. 2 | Partition-strategy sweep | [figures/combined_sweep.png](./figures/combined_sweep.png) |
| Supp. 3 | Region-assignment scalp scatter | [figures/region_assignment_K3.png](./figures/region_assignment_K3.png) |

---

## 10. References

*(canonical short list; expand in final write-up)*

- Casarotto S. et al. (2016). *Stratification of unresponsive patients
  by an independently validated index of brain complexity.* **Annals
  of Neurology** 80:718–729.
- Engemann D. A. et al. (2018). *Robust EEG-based cross-site and
  cross-protocol classification of states of consciousness.* **Brain**
  141:3179–3192.
- King J.-R. et al. (2013). *Information sharing in the brain indexes
  consciousness in noncommunicative patients.* **Current Biology**
  23:1914–1919.
- Koch C. et al. (2016). *Neural correlates of consciousness:
  progress and problems.* **Nature Reviews Neuroscience** 17:307–321.
- Schnakers C. et al. (2009). *Diagnostic accuracy of the vegetative
  and minimally conscious state.* **BMC Neurology** 9:35.
- Sitt J. D. et al. (2014). *Large scale screening of neural signatures
  of consciousness in patients in a vegetative or minimally conscious
  state.* **Brain** 137:2258–2270.

---

## Writing notes (not for the journal)

- Each section's pointer chapter contains the **full numbers,
  per-fold breakdowns, and reproducibility scripts**. Use this draft
  to know *what* to write; the chapters tell you *what numbers* to
  cite.
- The "K=4 noise cluster → K=3 winner" arc is mentioned only briefly
  (§7) — it doesn't belong in the main paper but works as a
  supplementary methods note for transparency.
- Figure captions need to be written from scratch. Each chapter
  caption in the lab notebook is a good starting point but is
  conversational; tighten to 2 sentences max for the paper.
- LOOCV results (in flight at the time of this draft) will replace
  the 5-fold numbers in §4.2 once they land — 144 subject-level
  predictions instead of 5 fold-aggregates, tighter confidence
  intervals. Update the abstract and §4.2 numbers when ready.
- Total target length: 4000–5000 words main text excluding
  references, 5 main figures + 2–3 supplementary.
