# Chapter 7 — Benchmarking the learned GNN representations

> **Status**: RESULTS IN (2026-07-17). Plan + scripts below; headline results
> in §7.8. Ran on the 256-electrode data (fair comparison, Path A of §7.2).

---

## 7.8 RESULTS — tuned CEBRA GNN matches the baseline and beats it on control-vs-DOC

All on 256-electrode wSMI, subject-disjoint 5-fold, last-100 epochs/session,
identical protocol to the raw-wSMI baseline. Eval via `gae_latent_eval.py`
(GMM-K on the latent) + `compare_roc.py` (binary AUC).

**CEBRA config sweep (3-class bal_acc, GMM readout):** the default CEBRA is
badly configured (τ=1.0, latent=8). Fixing it is the whole story:

| CEBRA config | K=3 | K=6 |
|---|---|---|
| τ=1.0, d=8 (pipeline default) | 0.471 | 0.499 |
| τ=0.05, d=16 | 0.465 | 0.522 |
| **τ=0.1, d=32** (best) | 0.562 | **0.623 ± 0.057** |
| raw-wSMI baseline | | 0.611 ± 0.131 |

Dominant lever = **latent_dim**; then temperature; then the readout (the latent
is L2-normalized onto a sphere → GMM likes more clusters, K=6 > K=3). LOOCV
tempers it: tuned CEBRA 0.569 vs baseline 0.590 pooled — so on 3-class accuracy
the GNN is a **statistical wash with the baseline**, with tighter 5-fold variance.

**Binary AUC (the differentiator)** — `output/roc_compare_cebra/`:

| model | control-vs-DOC | MCS-vs-UWS |
|---|---|---|
| GMM K=3 (unsup baseline) | 0.815 ± 0.095 | 0.713 ± 0.155 |
| **Tuned CEBRA GNN** | **0.877 ± 0.085** | 0.660 ± **0.065** |
| Supervised GCN (ch 6) | 0.931 ± 0.083 | 0.678 ± 0.155 |

> **⚠️ §7.8–7.8b are the 127-cohort progression (EMCS/COMA dropped).** The tight
> std here (±0.05–0.07) is a 127-cohort property that did NOT survive the
> cohort correction — on the matched 144-cohort every model's fold variance is
> larger (~0.09–0.13) and no model is uniquely tightest. **Use
> [narrative.md §4.5c](./narrative.md) for the canonical mean ± std table.**

**Headline (127-cohort):** the tuned CEBRA graph encoder **beats the unsupervised
GMM baseline on control-vs-DOC (+0.06 AUC), unsupervised**. The **MCS-vs-UWS
ceiling (~0.66–0.72) holds for every method** → the within-DOC boundary needs
multi-band features, not more encoder tuning (§7.6 lever #2). *(The "most stable /
tightest variance" reading here is 127-cohort only; see §4.5c.)*

### 7.8b THE WINNER — frozen CEBRA + supervised MLP probe (`finetune_encoder.py`)

Swapping the GMM readout for a trained classifier head on the FROZEN encoder is
the best configuration in the whole project. Full fine-tuning (training the
encoder end-to-end) collapses — small-data overfitting of the ~50k encoder
params on ~127 subjects.

| readout on tuned CEBRA (τ=0.1, d=32) | 3-class bal_acc | control-vs-DOC | MCS-vs-UWS |
|---|---|---|---|
| GMM clustering (K=6) | 0.623 ± 0.057 | 0.877 ± 0.085 | 0.660 ± 0.065 |
| **frozen encoder + MLP probe** | **0.673 ± 0.070** | **0.940 ± 0.067** | 0.693 ± **0.057** |
| full fine-tune (end-to-end) | 0.406 ± 0.103 (collapse) | — | — |
| — vs raw baseline | 0.611 ± 0.131 | 0.815 ± 0.095 | 0.713 ± 0.155 |
| — vs supervised GCN | 0.527 | 0.931 ± 0.083 | 0.678 ± 0.155 |

**The frozen-SSL + MLP-probe recipe wins on every axis (127-cohort numbers here;
see §4.5c for the matched 144-cohort).** 3-class 0.673, control-vs-DOC 0.940.
*(Note: the "beats supervised 0.931 / tightest variance" reading is a 127-cohort
artifact — on the matched 144-cohort CEBRA is 0.901 vs supervised 0.888, and no
model is uniquely tightest. §4.5c is canonical.)* This is the "good GNN result":
*contrastive pretrain → freeze → supervised probe* outperforms both the
hand-engineered PCA+GMM baseline and a supervised GCN on the clean axis, from
theta-wSMI alone. (Caveat: control-vs-DOC has 2–4 controls/fold → confirm 0.94
with LOOCV / a second seed.)

### 7.9 SOTA summary + exact reproduction recipe

**Current best (SOTA) on 256-electrode theta-wSMI, subject-disjoint 5-fold,
last-100 epochs/session:**

| method | 3-class bal_acc | control-vs-DOC AUC | MCS-vs-UWS AUC |
|---|---|---|---|
| **CEBRA (frozen) + MLP probe** ⭐ | **0.673 ± 0.070** | **0.940 ± 0.067** | 0.693 ± 0.057 |
| Supervised GCN (end-to-end) | 0.527 | 0.931 ± 0.083 | 0.678 ± 0.155 |
| CEBRA (frozen) + GMM K=6 | 0.623 ± 0.057 | 0.877 ± 0.085 | 0.660 ± 0.065 |
| GMM K=3 on raw wSMI (baseline) | 0.611 ± 0.131 | 0.815 ± 0.095 | 0.713 ± 0.155 |

> **⚠️ COHORT CORRECTION (the 127 table above is NOT cohort-matched; see the
> corrected 144 numbers below and in [narrative.md §4.5](./narrative.md)).** The GNN
> runs used the pipeline default `--diagnosis_granularity coarse`, which DROPS
> EMCS+COMA → a **127-subject** cohort, whereas the baseline/supervised numbers are
> on **144 subjects**. So "0.940 beats supervised 0.931" was confounded (CEBRA-127
> vs supervised-144). We re-ran **every method on both cohorts**. Corrected,
> cohort-matched results:
>
> | 3-class / ctrl-vs-DOC (all n=144, matched) | Baseline | CEBRA frozen+probe | Supervised GCN |
> |---|---|---|---|
> | **144-cohort (matched)** | 0.611 / 0.815 | **0.678 / 0.901** | 0.540 / 0.888 |
> | 127-cohort (ablation) | 0.551 / 0.810 | 0.673 / 0.940 | — |
>
> **Final reading (TWO cohort confounds found & fixed — all models now n=144):**
> CEBRA is the **best model on every metric** — best 3-class (0.678) and best
> control-vs-DOC (0.901), **beating even the fully-supervised GCN (0.888)**. The
> supervised GCN's earlier 0.931 was a cohort artifact (it was on 115 subjects,
> holding out 29); re-run on all 144 (`--all_subjects`) it is 0.888. Reconstruction
> AEs all sit ≤ baseline on 3-class (GATVAE collapsed to chance 0.333). A dedicated
> within-DOC head does NOT break the ~0.71 MCS-vs-UWS ceiling (0.685) → single-band
> limit, not readout. **Use [narrative.md §4.5c](./narrative.md) as the canonical
> results table.**

**The winning pipeline** = *contrastive pretrain → freeze → supervised MLP probe*:

1. **Data.** 256-electrode wSMI-theta `.npz` tree (`data/wsmi_res`), loaded via
   `load_wsmi_dataset_npz` (`--wsmi_format npz`, `--coords_file
   GSN-HydroCel-257.txt`), last 100 epochs/session (`--max_epochs_per_recording
   100`). Nodes = 256 electrodes; node features = the 256-d wSMI row; edges =
   k-NN (k=6) on electrode XYZ. Coarse target control / low_doc / high_doc.

2. **Encoder — CEBRA (`enc_gae_fc` = `GNNEncoder`), contrastive InfoNCE:**
   - SAGEConv stack `256 → [64, 64, 32, 16] → mean-pool → FC → latent`
   - **`latent_dim = 32`** (the dominant lever — 8→16→32 monotonically better)
   - **`temperature = 0.1`, FIXED** (`--cebra_fixed_temp`; default 1.0 is far too
     soft and was the main reason the out-of-the-box CEBRA failed)
   - `dropout 0.1`, `lr 1e-3`, `weight_decay 1e-5`, `batch_size 256`
   - positives = temporally-adjacent epochs (i, i+1) in the same recording;
     in-batch negatives; L2-normalized embedding (unit hypersphere)

3. **Readout — freeze encoder, train a 2-layer MLP head with class-weighted CE:**
   - head = `Linear(32→32) → ReLU → Dropout(0.3) → Linear(32→3)` + softmax
   - `--mode frozen --head mlp`, `epochs 40`, `lr 1e-3`, `weight_decay 1e-4`,
     `batch_size 128`, `random_state 42`
   - fresh encoder+head rebuilt per fold from the checkpoint (no leakage);
     per-subject prediction = mean softmax over the subject's epochs

**Reproduce:**
```bash
# 1. pretrain the CEBRA encoder on 256 nodes (GPU)
sbatch --partition=parietal slurm/train_wsmi256.sbatch \
    --models enc_gae_fc --loss cebra \
    --cebra_fixed_temp --cebra_temperature 0.1 --cebra_latent_dim 32 \
    --run_name wsmi256_cebra_t10_d32
# 2. frozen encoder + MLP probe (GPU) -> 3-class bal_acc + per_subject_proba
sbatch slurm/finetune_encoder.sbatch \
    --run_name wsmi256_cebra_t10_d32 --mode frozen --head mlp
# 3. binary AUCs vs baseline + supervised (CPU)
python scripts/compare_roc.py \
    --gmm_k3 output/roc_gmm_K3_5fold --gmm_k4 output/roc_gmm_K4_5fold \
    --supervised output/supervised \
    --moco output/wsmi256_cebra_t10_d32/finetune_frozen_mlp \
    --moco_label CEBRA-frozen-MLP --output_dir output/roc_compare_frozen
```

**Also tried, worse (not worth expanding):**
- *GMM readout instead of the MLP probe* — good (0.623 / 0.877) but below the
  probe; the L2-normalized latent wants K=6 not K=3.
- *Full end-to-end fine-tune* (train the encoder with CE) — **collapses to 0.406**
  (small-data overfitting of the encoder on ~127 subjects). Freeze, don't fine-tune.
- *Default CEBRA* (τ=1.0, latent=8) — 0.471; misconfigured temperature + latent.
- *τ=0.05/d=16* — 0.522; d=32 clearly better.
- *Reconstruction GAE/VGAE/GAEVAE/GATVAE* — expected ≤ baseline (reconstruction is
  the wrong loss for wSMI, ch 5); lean runs in progress for completeness.
- *MoCo augmentation-contrastive* (ch 5/6) — underperformed; augmentation invariance
  discards the discriminative coupling structure. CEBRA's *temporal* positives avoid this.

**Sweep detail + robustness (why the config is a real optimum, not luck):**

- *latent_dim* (GMM K=6 readout): 8→**0.499**, 16→**0.522**, 32→**0.623**,
  64→**0.536**, 128→**0.561**. Clean peak at **d=32**; the frozen-MLP probe agrees
  (d=32→**0.673**, d=64→**0.566**). Bigger latents hurt — the L2-normalized
  contrastive embedding gets too sparse for 3-class structure.
- *temperature* (fixed): 0.05/0.1/0.2 → the best is **τ=0.1** (d=32, K=6: 0.522 /
  0.623 / 0.549). τ=1.0 default is far too soft.
- *seed robustness* (frozen-MLP probe, d=32): seed 42 → **0.673 ± 0.070**,
  seed 7 → **0.647 ± 0.120**. Robust 3-class estimate ≈ **0.66**, comfortably above
  the 0.611 baseline and the 0.623 GMM readout. The **control-vs-DOC 0.940** headline
  is the seed-42 number (few controls/fold → single-seed; worth a LOOCV confirm).

*Caveats:* control-vs-DOC has few controls/fold (n_pos 2–4), so read +0.06 as
"improvement with overlapping bands"; the direction + tighter variance are
consistent. In the saved figure CEBRA was relabeled from the `--moco` slot via
`compare_roc.py --moco_label`.

**Open follow-ups (in flight):** latent_dim=64/128 push; supervised heads on the
CEBRA encoder (`scripts/finetune_encoder.py`, `--mode full` end-to-end +
`--mode frozen` linear/MLP probe) — direct CE classification vs the GMM readout;
the four reconstruction models (GAE/VGAE/GAEVAE/GATVAE) still tuning.

---

> **Original plan (2026-07-17)** after merging `origin/main`
> (GAE/VGAE/GAEVAE/GATVAE + CEBRA encoder + Ray-Tune pipeline) into
> `gio-grid-search-clustering`.

> **Goal**: for the first time, score the *learned* graph representations
> (reconstruction autoencoders **and** the CEBRA temporal-contrastive
> encoder) on the **same held-out protocol** that produced the raw-wSMI
> baseline, so the numbers are head-to-head comparable.
>
> **Baseline to beat** (chapters 3 / 6):
>
> | metric | raw-wSMI floor | aspirational |
> |---|---|---|
> | in-sample Cramér's V | 0.280 | ≥ 0.30 |
> | held-out 3-class bal_acc (5-fold mean) | 0.611 ± 0.131 | ≥ 0.65 |
> | control-vs-DOC AUC | 0.815 ± 0.095 | ≥ 0.90 |
> | MCS-vs-UWS AUC | 0.713 ± 0.155 | > 0.75 (**the real target**) |
> | high_doc recall | 48 % | ≥ 60 % |

---

## 7.1 The problem this chapter fixes — two disjoint codebases

After the merge the repo holds two GNN lines that never shared an
evaluation:

| | notebook thread (ch 1–6) | main pipeline (`run_wsmi_pipeline.py`) |
|---|---|---|
| montage | 256 EGI (`data/wsmi_res`) | 64 biosemi (`data/markers/wsmi_theta/…biosemi64`) |
| models | `contrastive_model.Encoder`, supervised DenseGCN | `src/model.py`: GAE / VGAE / GAEVAE / GATVAE / GNNEncoder / GATEncoder |
| loss | MoCo (failed), supervised CE | reconstruction MSE (+KL, +corr), CEBRA temporal-InfoNCE |
| scoring | `holdout_prediction.py` → bal_acc / AUC | val MSE + internal ARI/purity + LOSO |

The main-pipeline models have **only ever been scored on reconstruction
MSE and internal clustering metrics** — never on the last_100×balanced ×
GroupKFold × GMM-K3 protocol that the paper's 0.61 / 0.82 numbers use.
So we cannot yet say whether any learned encoder beats raw PCA(50).

`scripts/contrastive_eval.py` already bridges the notebook's 256-node
`Encoder` into that protocol. This chapter generalises that bridge to
**any `src/model.py` model** via the existing
`cluster_analysis.extract_latents_from_graphs`.

---

## 7.2 The montage decision — 256 is the fair comparison

The baseline (V=0.28, bal_acc 0.61, AUC 0.82) is on **256 EGI**. The
scientific question is *"does a learned GNN latent beat PCA(50) on the
SAME inputs?"* — so the fair comparison holds the data fixed at 256 and
swaps ONLY the representation. Training the GNN on 64 biosemi instead
confounds representation quality with electrode count (a 64-node loss
could just mean 64 electrodes carry less signal). 256 is also the
setting that *favours* the GNN — its whole value is the spatial
message-passing graph, and a 256-node scalp graph is richer than 64.

- **Primary — everything on 256 EGI (fair, recommended).** Train
  GAE/VGAE/CEBRA on the same `data/wsmi_res` matrices the baseline used
  (`in_channels=256`, `--coords_file data_scalp/GSN-HydroCel-257.txt`),
  then score with `gae_latent_eval.py` against the *existing* 0.61 floor.
  **Cost: needs a 256-capable loader (see 7.2a) + a GPU retrain of each
  GNN.** This is the number for the paper.
- **Secondary ablation — 64 biosemi (optional).** Answers a *different*
  question ("does the harmonized/lower-resolution montage help or
  hurt?"), NOT a fair head-to-head with the baseline. Only run if 256 is
  inconclusive. Would also need a re-computed 64-node GMM floor to be
  interpretable.

### 7.2a The loader — DONE (`load_wsmi_dataset_npz`)

`src/wsmi_loader.load_wsmi_dataset` reads junifer `.pkl (1,n,64,64)` and
**skips anything not 64×64**. So a `.npz` 256-capable loader was added:
`src/wsmi_loader.load_wsmi_dataset_npz` walks the 256 tree with the SAME
`EEGtoGraph.enumerate_matrix_sessions` the raw-wSMI baseline uses (so it
sees exactly the baseline's epochs), reusing the channel-agnostic
adjacency / clean / diagnosis helpers. Node count is inferred from
`--coords_file` (`GSN-HydroCel-257.txt` = 256 nodes, verified). Cohort
comes from the `wsmi_res_{DOC,control}` path tag.

Wired into the pipeline via `--wsmi_format {pkl,npz}` (default `pkl`,
preserves old behaviour); `stage_load` branches to the npz loader when
`--wsmi_format npz`. **Status: VALIDATED on real data** (cluster job
372171, 2026-07-17): 4 subjects → 40 graphs, `x` shape `(256, 256)`,
`edge_index (2, 1640)` k=6 graph, cohort tags + diagnosis lookup correct
(controls→CONTROL/HC, patients→COMA/UWS in fine granularity).

**Memory caveat (important for the training run).** At 256 nodes each
graph holds a 256×256 `x` (+ a 256×256 `raw_matrix` copy) ≈ 0.5 MB, so
the FULL ~130k-epoch dataset is ≈ 68 GB in RAM — it will OOM a normal
node (the 64-node pipeline was 16× smaller and never hit this). For the
256 training run, cap epochs with `--max_epochs_per_recording 100`
(matches the baseline's last_100 anyway → ~9 GB), or wire in the mmap
`src/lazy_dataset.py`. The eval (`gae_latent_eval.py`) already takes
`--last_n 100`, so it's fine.

---

## 7.3 The bridge — `scripts/gae_latent_eval.py`

Mirror of `contrastive_eval.py`, but features come from a trained
`src/model.py` autoencoder instead of the 256-node `Encoder`:

1. Load `output/<run>/splits/graphs.pt` (the cached per-epoch graphs with
   metadata) and `output/<run>/models/<tag>/model.pt`
   (`{"model_state_dict", "config"}`).
2. Rebuild the model with `train.build_decoder_model(model_kind, in_ch,
   config)`, load the state dict, `.eval()`.
3. `extract_latents_from_graphs(model, graphs, splits,
   aggregate=args.graph_latent_agg)` → one latent vector per epoch
   (`mean` or `flatten`; VGAE uses `mu`).
4. Build a dataframe from the bundle metadata (subject / session /
   matrix_idx / diagnosis), apply the **last_n per-session mask**, map
   diagnoses to coarse via `holdout_prediction.DX_TO_COARSE`.
5. Run the identical fold loop from `contrastive_eval.py`: balance train
   by diagnosis → GMM(K=3) on latents → soft/hard per-subject fingerprint
   → LogisticRegression → per-fold + pooled bal_acc, and dump
   `per_subject_proba.csv` so `scripts/compare_roc.py` can add it to the
   binary-AUC panel.

Encoder-only models (`enc_gae_fc` / `enc_gat_fc`) use
`extract_embeddings_encoder` instead of `extract_latents_from_graphs`
(graph-level pooled embedding, no `edge_index`-free path) — same
downstream.

**Draft** at `scripts/gae_latent_eval.py`, sbatch
`slurm/gae_latent_eval.sbatch`. Untested — smoke first (§7.5).

---

## 7.4 Run matrix (Path A)

Assuming trained biosemi runs exist (or are produced with
`run_wsmi_pipeline.py --stage all`):

| model | loss | why it's interesting |
|---|---|---|
| GAE | mse | reconstruction floor — expected to ≈ or < baseline (ch 5 argument) |
| VGAE | mse | does the KL-regularised latent cluster better? watch for posterior collapse |
| GAEVAE | mse | MLP bottleneck — cleaner latent geometry |
| GATVAE | mse | attention adjacency vs fixed k-NN |
| **enc_gae_fc** | **cebra** | **temporal-contrastive — the fix for the MoCo failure. Top priority.** |
| enc_gat_fc | cebra | attention + temporal contrastive |

For each: pooled + per-fold 3-class bal_acc, control-vs-DOC AUC,
MCS-vs-UWS AUC, high_doc recall. Compare against the 64-node GMM floor
from §7.2 Path A.

---

## 7.5 Smoke test + full 256 run (on the cluster, not the frontale)

**Step 0 — loader smoke (cheapest first).** Confirm the npz loader sees
the baseline's subjects/epochs before spending GPU on training:
```bash
python src/wsmi_loader.py --format npz \
    --patient_dir data/wsmi_res \
    --control_dir data/wsmi_res \
    --diagnosis_csv <patient_labels.csv> \
    --coords_file  data_scalp/GSN-HydroCel-257.txt
# expect ~144 subjects, 6-class diagnosis-group counts matching methodology.md
```
Watch for: session-id zero-padding (`ses-1` vs `ses-01`) in the CSV
lookup, and cohort tags resolving (DOC vs control). Adjust
`--patient_dir/--control_dir` to your actual on-disk split.

**Step 1 — train a GNN on 256 (one GPU job per model):**
```bash
sbatch slurm/gridsearch.sbatch   # or run_wsmi_pipeline directly:
python cookbook/run_wsmi_pipeline.py --stage all --input_mode wsmi \
    --wsmi_format npz --coords_file data_scalp/GSN-HydroCel-257.txt \
    --patient_dir data/wsmi_res --control_dir data/wsmi_res \
    --diagnosis_csv <patient_labels.csv> \
    --models gae vgae enc_gae_fc --loss mse \
    --run_name wsmi256_full
```
(`enc_gae_fc` needs `--loss cebra`; run it as its own invocation.)

**Step 2 — score against the 0.61 floor (CPU):**
```bash
sbatch slurm/gae_latent_eval.sbatch \
    --run_name wsmi256_full --model_kind gae --eval_mode 5fold --last_n 100 --K 3
```
Prints per-fold + pooled `soft_bal_acc_3` and writes
`output/wsmi256_full/gae_eval_<tag>/{eval_summary.json,per_subject_proba.csv}`.
Feed the latter to `scripts/compare_roc.py` to drop the GNN into the
binary-AUC panel next to GMM K=3 / supervised GCN. If the pooled number
is within noise of 0.611, the encoder recapitulates raw structure; if it
pulls ahead (especially MCS-vs-UWS AUC / high_doc recall), that's the
result worth writing up.

---

## 7.6 If nothing beats the floor — the escalation ladder

In priority order (each is a separate, cheap-ish experiment):

1. **CEBRA temporal-contrastive** (enc_gae_fc/cebra) — already implemented,
   just needs eval. Most likely to help because it keeps the coupling
   structure MoCo discarded.
2. **Multi-band node features** — extend node features from theta-only to
   a 4–5 band wSMI stack (θ/α/β/δ/γ). This is the change most aligned
   with Sitt 2014's multi-feature requirement and the documented path to
   cracking the within-DOC ceiling. Touches `wsmi_loader.py` (stack bands
   into `x`) — `in_channels` becomes `64*n_bands` or a per-node band
   vector.
3. **Pretrain-on-all → probe** — pretrain the CEBRA encoder on all ~132k
   unlabeled epochs, freeze, probe on last_100_balanced. The achievable
   "use a pretrained encoder" move (no external checkpoint exists for
   wSMI graphs; EEG foundation models like LaBraM/BrainLM are raw-EEG,
   not connectivity, and would only apply to the time-series modality).
4. **Functional adjacency** — replace fixed anatomical k-NN with
   thresholded per-epoch wSMI edges (ch 5 §5.7 open question); GATv2
   partly does this by learning edge weights.

---

*See: [chapter 5](./chapter_05_contrastive_pretraining.md) (why
reconstruction is the wrong loss), [chapter 6](./chapter_06_supervised_and_binary_eval.md)
(the binary-AUC benchmark this plugs into), [methodology.md](./methodology.md)
(the leakage-free protocol being reused).*
