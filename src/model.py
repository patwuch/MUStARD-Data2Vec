"""Sarcasm classification model.

A single flexible class handles all ablation configurations from the paper:
  - n_modalities: 1 (unimodal), 2 (bimodal), or 3 (trimodal)
  - use_context: whether context embeddings are provided
  - use_speaker: whether speaker one-hot embedding is concatenated

Architecture follows MUStARD++ with collaborative cross-modal attention gates.
Modality positions A/B/C are assigned by the caller (see train_utils.MODALITY_POSITIONS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SarcasmClassifier(nn.Module):
    def __init__(
        self,
        n_modalities: int,
        use_context: bool,
        use_speaker: bool,
        n_speaker: int = 0,
        input_dim: int = 768,
        shared_dim: int = 768,
        proj_dim: int = 768,
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()

        self.n_modalities = n_modalities
        self.use_context = use_context
        self.use_speaker = use_speaker
        self.shared_dim = shared_dim

        pos_labels = "ABC"[:n_modalities]
        for pos in pos_labels:
            setattr(self, f"{pos}_utt_proj", nn.Linear(input_dim, shared_dim))
            setattr(self, f"{pos}_utt_norm", nn.BatchNorm1d(shared_dim))
            if use_context:
                setattr(self, f"{pos}_ctx_proj", nn.Linear(input_dim, shared_dim))
                setattr(self, f"{pos}_ctx_norm", nn.BatchNorm1d(shared_dim))

        # Collaborative attention gates (shared across all modality pairs)
        self.gate1 = nn.Linear(2 * shared_dim, proj_dim)
        self.gate2 = nn.Linear(proj_dim, shared_dim)

        speaker_dim = n_speaker if use_speaker else 0
        pred_in = n_modalities * shared_dim + speaker_dim

        self.pred_module = nn.Sequential(
            nn.Linear(pred_in, 2 * shared_dim),
            nn.BatchNorm1d(2 * shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _project(self, pos: str, tensor: torch.Tensor, is_context: bool) -> torch.Tensor:
        if is_context:
            return getattr(self, f"{pos}_ctx_norm")(
                F.relu(getattr(self, f"{pos}_ctx_proj")(tensor))
            )
        return getattr(self, f"{pos}_utt_norm")(
            F.relu(getattr(self, f"{pos}_utt_proj")(tensor))
        )

    def _pairwise_attention(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate1(torch.cat((query, key), dim=1)), dim=1)

    def _attend(self, query: torch.Tensor, *keys: torch.Tensor) -> torch.Tensor:
        combined = sum(self._pairwise_attention(query, k) for k in keys)
        return F.softmax(self.gate2(combined), dim=1)

    def forward(
        self,
        uA: torch.Tensor,
        uB: torch.Tensor = None,
        uC: torch.Tensor = None,
        cA: torch.Tensor = None,
        cB: torch.Tensor = None,
        cC: torch.Tensor = None,
        speaker_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        u_inputs = {"A": uA, "B": uB, "C": uC}
        c_inputs = {"A": cA, "B": cB, "C": cC}
        pos_labels = "ABC"[: self.n_modalities]

        u_proj = {p: self._project(p, u_inputs[p], False) for p in pos_labels}
        c_proj = (
            {p: self._project(p, c_inputs[p], True) for p in pos_labels}
            if self.use_context
            else {}
        )

        # All features available for cross-modal attention
        all_features = []
        for p in pos_labels:
            all_features.append(u_proj[p])
            if self.use_context:
                all_features.append(c_proj[p])

        # Update each utterance representation via attention over all others
        updated = {}
        for p in pos_labels:
            others = [f for f in all_features if f is not u_proj[p]]
            if others:
                updated[p] = u_proj[p] * self._attend(u_proj[p], *others)
            else:
                updated[p] = u_proj[p]

        parts = [updated[p] for p in pos_labels]
        if self.use_speaker and speaker_embedding is not None:
            parts.append(speaker_embedding)

        return self.pred_module(torch.cat(parts, dim=1))
