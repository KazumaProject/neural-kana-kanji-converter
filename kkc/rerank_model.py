from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RerankTransformerConfig:
    vocab_size: int
    max_len: int = 256
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    ffn: int = 1024
    dropout: float = 0.1


class RerankTransformer(nn.Module):
    """
    Cross-encoder style reranker.
    Input: a single sequence that concatenates (src + SEP + candidate) at char level.
    Output: a scalar score (higher is better).

    Pooling: use the hidden state of the first token (BOS).
    """
    def __init__(self, cfg: RerankTransformerConfig, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ffn,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.dropout = nn.Dropout(cfg.dropout)
        self.scorer = nn.Linear(cfg.d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.scorer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.scorer.bias)

    def forward(
        self,
        input_ids: torch.Tensor,                 # (B, L)
        key_padding_mask: Optional[torch.Tensor] # (B, L) bool, True for PAD
    ) -> torch.Tensor:
        """
        Returns:
            scores: (B,) float32
        """
        bsz, seqlen = input_ids.shape
        if seqlen > self.cfg.max_len:
            input_ids = input_ids[:, : self.cfg.max_len]
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, : self.cfg.max_len]
            seqlen = input_ids.size(1)

        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, -1)  # (B, L)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, L, D)

        pooled = h[:, 0, :]  # BOS pooling
        scores = self.scorer(pooled).squeeze(-1)
        return scores
