# Paper narrative — unsupervised and self-supervised graph representations of consciousness from resting-state theta-wSMI

> **Status (2026-07-17):** working narrative that structures the full paper and
> records every method exactly. Numbers are drawn from the lab-notebook chapters
> (raw-wSMI baseline) and the chapter-7 GNN experiments, all verified against the
> code (see `§0 Provenance & caveats`). **One correction is in flight:** the GNN
> runs were trained on a 127-subject cohort (EMCS+COMA silently dropped by the
> loader's default `coarse` granularity); a cohort-matched 144-subject re-run
> (`--diagnosis_granularity fine`, SLURM job 373415) is running and the GNN
> headline AUCs will be updated from it. GNN numbers below are flagged
> **[127-cohort, preliminary]** until then.

---

## 0. Provenance & caveats (read first — these gate every claim)

1. **The wSMI matrices are computed UPSTREAM, not in this repo.** The 256-electrode
   theta-band wSMI was computed by "Pablo's group" and redistributed by Greta via
   Google Drive on **2026-05-26** (`methodology.md`). The repo *consumes*
   per-session `.npz` files of shape `(n_epochs, 256, 256)`. Therefore the exact
   wSMI hyperparameters for the 256 data — symbol length (kernel), the temporal
   spacing τ, and the sampling rate — are **not recorded in this repository** and
   must be obtained from the upstream provider before final write-up. Do **not**
   state "kernel=3" as repo-verified. (The *separate* biosemi64 variant, not used
   for the main results, does expose τ=10 for resting-state and sfreq=100 Hz.)
2. **Node count is 256, not 257.** `data_scalp/GSN-HydroCel-257.txt` contains 256
   electrode rows (E1–E256, no fiducials); "257" is the montage name.
3. **Two coarse-label mappings exist in the codebase** and they differ — see §3.1.
   The *baseline* keeps all six CRS-R classes and coarsens to
   control / low_doc={UWS,COMA} / high_doc={MCS−,MCS+,EMCS} (144 subjects). The
   *GNN loader's default coarse mode* drops EMCS and COMA (127 subjects). The
   cohort-matched GNN re-run uses `fine` granularity to restore the baseline's
   144-subject cohort.
4. **`methodology.md` has a stray typo** ("5 s recording → 6.25 epochs") that
   contradicts the consistently-stated 0.8 s epoch length and ~640–1,200
   epochs/session; use 0.8 s.

---

## 1. Framing

### 1.1 Clinical problem
Distinguishing **unresponsive wakefulness syndrome (UWS,** formerly vegetative
state**)** from the **minimally conscious state (MCS)** at the bedside is hard: the
gold-standard behavioural scale (CRS-R) has a documented **~40 % misdiagnosis
rate** (Schnakers et al. 2009). EEG-based biomarkers are the most scalable
adjunct.

### 1.2 The signal
**Weighted Symbolic Mutual Information (wSMI)** (King et al. 2013) quantifies
non-linear information sharing between two EEG channels by symbolising each
signal and computing a weighted mutual information that discounts trivially
similar symbols (volume conduction). Large supervised studies converge on
**theta-band wSMI between centro-posterior electrodes** as the single most
discriminative EEG feature of consciousness (Sitt et al. 2014, *Brain*;
Engemann et al. 2018, *Brain*). We use **theta-band wSMI only**.

### 1.3 Contributions
1. **Unsupervised replication.** Clustering raw theta-wSMI (no labels) recovers
   the published consciousness signature — PC1 of the feature space orders the
   diagnoses monotonically, and parieto-occipital coupling is the most
   consciousness-sensitive, matching Sitt/King/Casarotto — and generalises to
   unseen patients (held-out 3-class balanced accuracy ≈ 0.59–0.61).
2. **A learned self-supervised graph representation.** A contrastive (CEBRA-style)
   graph encoder, **frozen**, with a lightweight supervised MLP probe, is the
   best representation we found: on the conscious/unconscious axis it matches or
   exceeds both the hand-engineered baseline and a fully-supervised GCN
   **[127-cohort, preliminary — cohort-matched re-run in flight]**.
3. **A hard ceiling within DOC.** No method (unsupervised, supervised, or
   self-supervised) exceeds ~0.66–0.72 AUC on MCS-vs-UWS from theta-wSMI alone,
   pinpointing multi-band features as the necessary next ingredient.

---

## 2. Data & signal (verified)

### 2.1 Cohort
- **144 unique subjects** = **130 DOC patients + 14 healthy controls**.
- The DOC set was **132 patients / 164 sessions** before dropping **2 patients**
  whose `diagnostic_crs_final` was missing → **130 DOC**.
- **178 sessions total** = 164 DOC sessions + 14 control sessions (1/control).
- Controls are identified by **numeric subject IDs** (`sub-001…sub-014`); patients
  have alphanumeric IDs.
- CRS-R diagnoses (`diagnostic_crs_final`): **UWS, MCS−, MCS+, EMCS, COMA** (+
  control by construction).

### 2.2 EEG acquisition & montage
- **256-channel EGI HydroCel** net; **256 usable electrodes** after reference drop.
- The learned-graph pipeline builds a fixed spatial graph over these 256
  electrodes (§3.3). A separate 64-channel biosemi montage exists in the repo but
  is **not** used for the main results.

### 2.3 wSMI signal
- **Theta band, weighted Symbolic Mutual Information**, one **(256, 256)** symmetric
  connectivity matrix **per 0.8 s epoch**, zero diagonal, values roughly in
  `[0, 0.3]`.
- Sessions contain ~640–1,200 epochs (~10-min recordings).
- **132,041 epochs total** entering analysis. Per-diagnosis epoch counts
  (verified to sum to 132,041): UWS 56,584; MCS− 31,903; MCS+ 20,569; EMCS 11,505;
  COMA 6,271; control 5,209.

### 2.4 Prediction target
The learning target is the **subject's diagnosis group**. The primary problem is
the **coarse 3-class** collapse the clinic cares about:
**control / low_doc / high_doc**, where (baseline mapping)
low_doc = {UWS, COMA} (unconscious) and high_doc = {MCS−, MCS+, EMCS} (minimally
conscious / emerging). We also report the two clinically meaningful **binary**
tasks: **control-vs-any-DOC** and **MCS-vs-UWS (high_doc vs low_doc)**. Prediction
is **per-subject**, aggregated from per-epoch features.

---

## 3. Methods (exact)

### 3.1 Two coarse-label mappings (must be reconciled in the paper)
- **Baseline / cohort-matched target:** control; low_doc = {UWS, VS, COMA};
  high_doc = {MCS−, MCS+, EMCS}. Keeps all 144 subjects. Unknown-dx dropped only.
- **GNN loader default (`coarse`):** UWS/VS→UWS, MCS−/MCS+→MCS, **EMCS & COMA →
  DROP**. This silently reduced the GNN cohort to **127 subjects** (14 control,
  53 UWS, 60 MCS). The **`fine`** granularity keeps EMCS/COMA (only VS→UWS); the
  GNN eval then re-coarsens with the baseline mapping, restoring 144 subjects.
  **All GNN numbers below marked [127-cohort] must be replaced by the fine-run.**

### 3.2 Unsupervised raw-wSMI baseline (the paper's reference method)
Per-epoch (256,256) matrix → **upper triangle** `triu_indices(256, k=1)` =
**32,640-dim** vector. Then, **fit on training subjects only** inside each CV fold:
1. `StandardScaler` (z-score each of the 32,640 features).
2. `PCA(n_components=50, svd_solver='randomized', random_state=42)`.
3. `GaussianMixture(n_components=K, covariance_type='full', n_init=3,
   max_iter=200, reg_covar=1e-3, random_state=42)`. **K=3** is the paper default
   (chosen over K=4, §4.3); `reg_covar=1e-3` was set after a singleton cluster
   appeared at 1e-4.
4. **Epoch selection: last 100 epochs per session** (`last_n_per_session_mask`,
   ~80 s at the end of each recording — empirically the most discriminative
   window, §4.3).
5. **Diagnosis balancing (train only):** down-sample each diagnosis to the
   smallest class count before fitting the GMM (seeded RNG). Balanced last_100
   subset ≈ **4,200 epochs**.

**Per-subject prediction (three strategies):**
- (A) modal cluster → training-fold majority-diagnosis lookup;
- (B) **hard fingerprint** = K-dim vector of cluster-occupancy fractions over the
  subject's last_100 epochs → `LogisticRegression`;
- (C) **soft fingerprint** = mean GMM posterior (`predict_proba`) over the
  subject's epochs → `LogisticRegression`.

`LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs',
random_state=42)`. Metric: `balanced_accuracy_score`. Soft-FP is the headline
readout.

### 3.3 Graph construction (shared by all GNN models)
- **Nodes = 256 electrodes.** **Node features** `x` = that electrode's full row of
  the (256,256) wSMI matrix (256-dim connectivity profile). The matrix is
  symmetrised, NaN/Inf→0, diagonal→0.
- **Edges = spatial k-nearest-neighbour graph**, `k=6`, Euclidean distance on
  electrode **(x,y,z)** coordinates, symmetrised and **binarised**
  (`A = (A+Aᵀ) > 0`). Fixed across all epochs/subjects — the graph encodes scalp
  topology, **not** wSMI values. For 256 nodes this yields `edge_index (2, 1640)`.
- Critically, wSMI enters only as node features, never as edge weights — so the
  model must learn coupling structure from features over a fixed anatomical graph.

### 3.4 Self-supervised graph encoder — CEBRA-style contrastive (`enc_gae_fc`)
- **Encoder (`GNNEncoder`):** SAGEConv stack
  `256 → [64, 64, 32, 16]` (each `SAGEConv(aggr='mean', project=True)` +
  BatchNorm1d + LeakyReLU(0.1) + Dropout) → **`global_mean_pool`** (one vector per
  epoch-graph) → FC head `Linear(16→16) → LeakyReLU(0.1) → Linear(16→latent_dim)`
  → **L2-normalisation** (embeddings on the unit hypersphere). No decoder.
- **Objective — InfoNCE (CEBRA-style), temporal positives.** For each epoch *i*
  the **positive** is epoch *i+1* in the **same recording** (grouped by
  (subject, session, acq), ordered by `matrix_idx`); negatives are the other
  positives in the batch; similarity is cosine over L2-normalised embeddings.
  Temperature is learnable by default (init 1.0, floored at 0.1); the **winning
  config fixes it** (`--cebra_fixed_temp --cebra_temperature 0.1`).
- **Training (`stage_train_cebra`):** `AdamW(lr=1e-3, weight_decay=1e-5)`,
  `ReduceLROnPlateau(factor=0.5, patience=15)`, up to 150 epochs, early stopping
  (patience 25) on **validation InfoNCE**, best-val checkpointing. batch_size 256,
  dropout 0.1.
- **Winning hyperparameters:** `latent_dim = 32`, `temperature = 0.1` (fixed).
  These are a *characterised optimum* (§4.5), not a lucky point.

### 3.5 Readouts on the encoder (the key comparison)
Given the frozen/trained encoder embedding per epoch:
- **(i) GMM clustering readout** — `gae_latent_eval.py`: swap the baseline's
  PCA(50) features for the encoder embedding, keep the *identical* GMM-K +
  fingerprint + LogReg protocol. Best at **K=6** (the L2-normalised spherical
  latent wants more clusters than K=3).
- **(ii) Frozen encoder + supervised MLP probe** — `finetune_encoder.py
  --mode frozen --head mlp`: freeze the encoder, train
  `Linear(latent→32) → ReLU → Dropout(0.3) → Linear(32→3)` with class-weighted
  cross-entropy (AdamW lr 1e-3, wd 1e-4, 40 epochs, batch 128); per-subject
  prediction = mean softmax over epochs; fresh encoder+head rebuilt per fold.
- **(iii) Full fine-tune** — `--mode full`: same head, but **all encoder weights
  trained** end-to-end with CE.

### 3.6 Supervised GCN baseline (chapter 6)
An end-to-end DenseGCN over the same anatomical graph with a 3-class
cross-entropy head, trained subject-disjoint (chapter 6). Serves as the
"supervised upper reference."

### 3.7 Evaluation protocol (leakage-safe)
- **Subject-disjoint `GroupKFold` (5-fold)**; every epoch of a subject is in
  exactly one fold. Also **LOOCV** (one subject per fold, pooled) for the tightest
  estimate. All preprocessors (scaler, PCA, GMM, encoder, LogReg/MLP) fit on
  **training subjects only**.
- **Metrics:** 3-class **balanced accuracy** (chance 0.333); **binary ROC AUC**
  with 1,000-sample bootstrap CIs for control-vs-DOC and MCS-vs-UWS.
- **Reproducibility:** `random_state=42` throughout (a second seed, 7, used for
  stability checks). Balancing applied to train only, never test.

---

## 4. Results

### 4.1 Method ablation (chapter 1)
On raw theta-wSMI (StandardScaler→PCA(50)→cluster), **GMM (full covariance) beats
KMeans, Spectral, and Louvain**. Full-data Cramér's V (cluster×diagnosis):
GMM K=4 **0.183** vs KMeans **0.148**; Spectral/Louvain ≤0.11. Density/probabilistic
clustering handles the high-dimensional, elliptical wSMI structure better than
graph-based methods. PCA(50) captures only ~11.6 % of variance on full data
(~32 % on last_100) — the signal is high-dimensional and partly non-linear,
motivating a learned representation.

### 4.2 Interpretability — the consciousness gradient is the principal axis (chapter 2)
- **PC1 of the standardised upper-triangle wSMI orders the diagnoses strictly
  monotonically**, unsupervised. Per-diagnosis subject-mean PC1 (from
  `pc1_interpretation`): control **+24.15**, EMCS +2.59, MCS+ −1.76, MCS− −4.83,
  UWS −7.18, COMA **−10.83**. The **control→EMCS jump (~22 units)** is ~5× any
  within-DOC step — the conscious/unconscious boundary is qualitatively larger
  than within-DOC gradations. PC1 explains only ~4.2 % of variance yet carries the
  discriminative ordering; PCA(2) and UMAP(2) both recover it. **This is the
  "1-parameter consciousness manifold" the project set out to find, and it falls
  out of raw wSMI with no labels.**
- **Region decomposition:** grouping the 256 electrodes into 5 regions (F/C/P/T/O
  by coordinate thresholds), the top PC1 loadings are all positive
  (more coupling = more conscious) and dominated by **posterior cortex**: P–O
  (+0.0065), C–O, C–P, O–O, P–P lead.

### 4.3 Data engineering + held-out validation (chapter 3)
- **Two independent levers stack:** last-100-epochs/session (V 0.183→0.219) +
  diagnosis-balancing before GMM fit (→0.252). **K=3** beats K=4 on held-out
  generalisation and centroid cleanliness (K=4's 4th cluster was a 53-epoch noise
  sliver). In-sample **V = 0.28** (K=3).
- **Held-out (subject-disjoint) 3-class balanced accuracy:**
  **5-fold 0.611 ± 0.131** (soft-FP, K=3); **LOOCV pooled 0.590**. vs chance 0.333.
- **Confusion structure (LOOCV, K=3):** the conscious/unconscious boundary is
  clean (4 cross-errors / 75 control↔low_doc pairs); **high_doc (MCS) recall only
  ~48 %**, errors spread to both neighbours — the within-DOC boundary is the hard
  part, mirroring the CRS-R misdiagnosis rate.

### 4.4 Regional replication of the literature (chapter 4)
Ranking the 15 region-pairs by spread across the K=3 clusters: **P–O is the most
consciousness-sensitive (0.018)**, followed by P–P, C–P, C–O, O–O — all posterior.
**T–T is essentially flat (0.0015)** — temporal coupling does not track
consciousness. Every non-T–T pair has stronger coupling in the conscious cluster.
This reproduces Sitt 2014 (centro-posterior), King 2013 (long-range
posterior→frontal), and Casarotto 2016 (posterior hot zone) — **without labels.**

### 4.5 The learned representation — GNN results **[127-cohort, preliminary]**
> ⚠️ These numbers were computed on the 127-subject cohort (EMCS+COMA dropped by
> the loader default). The **cohort-matched 144-subject re-run (job 373415)** is
> in flight; replace the AUCs below with its output. The *qualitative* findings
> (relative ordering, optima) are expected to hold.

**Readout comparison on the tuned CEBRA encoder (latent_dim=32, τ=0.1):**

| readout | 3-class bal_acc | control-vs-DOC AUC | MCS-vs-UWS AUC |
|---|---|---|---|
| raw-wSMI baseline (144-subj) | 0.611 ± 0.131 | 0.815 ± 0.095 | 0.713 ± 0.155 |
| CEBRA frozen + GMM K=6 | 0.623 ± 0.057 | 0.877 ± 0.085 | 0.660 ± 0.065 |
| **CEBRA frozen + MLP probe** | **0.673 ± 0.070** | **0.940 ± 0.067** | 0.693 ± 0.057 |
| CEBRA full fine-tune | 0.406 ± 0.103 (collapse) | — | — |
| Supervised GCN (144-subj) | 0.527 | 0.931 ± 0.083 | 0.678 ± 0.155 |

Qualitative findings (robust):
- **Freeze, don't fine-tune.** The frozen encoder + trained MLP head is the best
  readout; **full end-to-end fine-tuning collapses** (0.41) — small-data
  overfitting of ~50k encoder params on ~127 subjects.
- **A trained probe beats clustering the representation** (0.673 vs 0.623 GMM).
- On control-vs-DOC the frozen probe reaches the top of the field
  **[cohort-matched value pending]**; on MCS-vs-UWS it sits at the universal
  ceiling but with the tightest variance.

**Characterised optimum (127-cohort ablations, robust):**
- **latent_dim** (GMM K=6): 8→0.499, 16→0.522, **32→0.623**, 64→0.536, 128→0.561 —
  a clean interior peak at **32**; the frozen probe agrees (d=32→0.673, d=64→0.566).
- **temperature** (fixed): 0.05/0.1/0.2 → **0.1** best (0.522/0.623/0.549);
  default 1.0 is far too soft and was why out-of-the-box CEBRA failed (0.471).
- **seed robustness** (frozen probe, d=32): seed 42 → 0.673, seed 7 → 0.647 →
  robust ~0.66.

### 4.6 The within-DOC ceiling (all methods)
Every method — unsupervised GMM (0.713), supervised GCN (0.678), CEBRA frozen
(0.693) — lands at **~0.66–0.72 AUC on MCS-vs-UWS**, with large fold variance and
complementary failure modes. From theta-wSMI alone the within-DOC boundary is not
crackable; this matches Sitt 2014's requirement of a **multi-band, multi-feature**
panel (θ+α+β+δ+γ) to exceed ~0.78.

---

## 5. Discussion
1. **The consciousness signal lives in the raw data, not just the supervised loss
   surface.** PC1 of raw wSMI already orders diagnoses correctly; clustering tiles
   this manifold; a learned encoder sharpens it.
2. **Posterior-centric coupling is the universal marker** — recovered three ways
   (GMM centroid contrast, PC1 loadings, region-pair sensitivity), all pointing at
   the parieto-occipital hot zone, label-free.
3. **Self-supervision + a light probe is the sweet spot** for this small-N regime:
   contrastive pretraining learns a stable representation from all epochs, and a
   frozen probe avoids the overfitting that sinks end-to-end fine-tuning.
4. **The intermediate/within-DOC state is the open frontier** — flat regional
   signature, ~0.7 AUC ceiling — and is where multi-band features should help.

## 6. Limitations
- **Single band (theta).** The likely reason MCS-vs-UWS is capped.
- **Upstream wSMI parameters not in-repo** (§0.1) — must be documented from source.
- **Small extremes:** 14 controls, 11 EMCS, 7 COMA → binary AUCs with few
  positives/fold are noisy (esp. control-vs-DOC).
- **GNN cohort correction pending** (§0.3) — headline AUCs to be finalised on the
  144-subject matched run.
- **Single dataset / single site** (Paris cohort). External replication needed.
- **Coordinate-driven 5-region parcellation** is coarse.

## 7. Future work
1. **Multi-band node features** (θ+α+β+δ+γ wSMI stacked per node) — the top lever
   for the within-DOC ceiling.
2. **Outcome prediction:** `patient_labels.csv` carries `cs_{6m,1y,2y}` recovery
   outcomes — does baseline representation predict *recovery*, a stronger clinical
   test than current-state classification?
3. **Cross-site validation** on the Engemann 2018 cohorts.
4. **Pretrain-on-all → probe** at scale; functional (wSMI-derived) adjacency;
   attention (GATv2) encoders.

## 8. Reproducibility
- **Data:** 256-electrode theta-wSMI `.npz` tree (`data/wsmi_res`), loaded via
  `load_wsmi_dataset_npz` (`--wsmi_format npz --coords_file
  data_scalp/GSN-HydroCel-257.txt`), last-100 epochs/session,
  **`--diagnosis_granularity fine`** to keep all 144 subjects.
- **Baseline:** `scripts/holdout_prediction.py` (K=3, soft-FP).
- **SOTA GNN:** pretrain `slurm/train_wsmi256.sbatch --models enc_gae_fc --loss
  cebra --cebra_fixed_temp --cebra_temperature 0.1 --cebra_latent_dim 32`; probe
  `slurm/finetune_encoder.sbatch --mode frozen --head mlp`; binary AUC
  `scripts/compare_roc.py`.
- **Seeds:** 42 (primary), 7 (stability). All fits train-only per fold.
- Full recipe + exact hyperparameters: `chapter_07 §7.9`.

## 9. Figure plan (from paper_draft §9, updated)
1. Pipeline schematic (wSMI → graph → encoder → probe / GMM).
2. PC1 consciousness gradient (PCA(2) by diagnosis + PC1 boxplot).
3. Region-pair consciousness-sensitivity 5×5 + posterior-hot-zone scalp.
4. Held-out 3-class confusion matrix (baseline, LOOCV).
5. **Binary ROC panel** (control-vs-DOC & MCS-vs-UWS) across baseline / GMM-readout
   / frozen-probe / supervised — **cohort-matched version**.
6. CEBRA ablations (latent_dim + temperature sweeps; frozen vs full fine-tune).

## 10. References
Sitt 2014 (*Brain*); King 2013 (*Curr Biol*); Casarotto 2016 (*Ann Neurol*);
Engemann 2018 (*Brain*); Koch 2016 (*Nat Rev Neurosci*); Schnakers 2009
(*BMC Neurol*); Schrimpf/CEBRA (Schneider et al. 2023, *Nature*); He 2020 (MoCo).

---

*Sources: lab-notebook chapters 1–7 + methodology; code verified 2026-07-17
(see §0). GNN headline AUCs pending the cohort-matched re-run (job 373415).*
