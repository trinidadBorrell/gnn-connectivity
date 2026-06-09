"""
CEBRA-STYLE CONTRASTIVE LOSS (in-repo InfoNCE)
==============================================
A lightweight re-implementation of CEBRA's InfoNCE objective
(https://cebra.ai/) for the encoder-only `GNNEncoder` (`enc_gae_fc`). We do not
depend on the `cebra` package: the only reusable piece is the criterion, while
its conv encoders / solver / data loaders assume fixed receptive-field models
over raw arrays and do not accept PyG graphs.

Contrastive structure (time-contrastive):
- one embedding per epoch-graph,
- the POSITIVE of reference epoch i is epoch i+1 in the SAME recording
  (subject+session+acq, ordered by matrix_idx) — epochs are temporally ordered,
- NEGATIVES are the other positives in the batch (standard in-batch negatives),
  with the reference's own positive excluded from its negative set.

Embeddings are assumed L2-normalized (GNNEncoder does this), so the similarity is
cosine. Temperature follows CEBRA: optionally learnable, floored at `min_temp`.
"""
from collections import defaultdict
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class InfoNCE(nn.Module):
    """CEBRA-style InfoNCE with cosine similarity and (optional) learnable temperature.

    Args:
        temperature: initial temperature.
        learn_temperature: if True, temperature is a trained parameter.
        min_temp: lower floor on temperature (kept >= this).
    """
    def __init__(self, temperature: float = 1.0, learn_temperature: bool = True,
                 min_temp: float = 0.1):
        super().__init__()
        self.min_temp = float(min_temp)
        init = max(float(temperature) - self.min_temp, 1e-4)
        # softplus(raw) = init  ->  temp = min_temp + softplus(raw) ~= temperature
        raw_init = torch.log(torch.expm1(torch.tensor(init)))
        if learn_temperature:
            self._temp_raw = nn.Parameter(raw_init)
        else:
            self.register_buffer("_temp_raw", raw_init)

    @property
    def temperature(self) -> torch.Tensor:
        return self.min_temp + F.softplus(self._temp_raw)

    def forward(self, ref: torch.Tensor, pos: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the InfoNCE loss with in-batch negatives.

        Args:
            ref: (B, d) reference embeddings (already L2-normalized).
            pos: (B, d) positive embeddings (the temporal successor of each ref).

        Returns:
            (loss, alignment) where alignment is the mean positive cosine
            similarity (undivided by temperature) for logging.
        """
        temp = self.temperature
        pos_sim = (ref * pos).sum(dim=-1, keepdim=True) / temp   # (B, 1)
        neg_sim = (ref @ pos.t()) / temp                          # (B, B)
        # Exclude each reference's own positive from its negatives.
        eye = torch.eye(ref.size(0), dtype=torch.bool, device=ref.device)
        neg_sim = neg_sim.masked_fill(eye, float("-inf"))
        logits = torch.cat([pos_sim, neg_sim], dim=1)             # (B, 1+B), pos at col 0
        target = torch.zeros(ref.size(0), dtype=torch.long, device=ref.device)
        loss = F.cross_entropy(logits, target)
        alignment = (ref * pos).sum(dim=-1).mean()
        return loss, alignment


class CebraPairDataset(Dataset):
    """Temporal (ref, pos) pairs for time-contrastive training.

    Groups graphs by recording (subject_id, session_num, acq), orders them by
    matrix_idx, and links each epoch to its successor. The last epoch of each
    recording has no successor and is never used as a reference.
    """
    def __init__(self, graphs: Sequence):
        self.graphs = graphs
        groups = defaultdict(list)
        for idx, g in enumerate(graphs):
            key = (getattr(g, "subject_id", None),
                   getattr(g, "session_num", None),
                   getattr(g, "acq", None))
            groups[key].append(idx)
        pairs: List[Tuple[int, int]] = []
        for key, idxs in groups.items():
            idxs_sorted = sorted(idxs, key=lambda i: int(getattr(graphs[i], "matrix_idx", 0)))
            for a, b in zip(idxs_sorted[:-1], idxs_sorted[1:]):
                pairs.append((a, b))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i):
        ref_idx, pos_idx = self.pairs[i]
        return self.graphs[ref_idx], self.graphs[pos_idx]


def collate_pairs(batch):
    """Collate (ref_graph, pos_graph) tuples into two PyG Batches."""
    refs = [b[0] for b in batch]
    poss = [b[1] for b in batch]
    return Batch.from_data_list(refs), Batch.from_data_list(poss)
