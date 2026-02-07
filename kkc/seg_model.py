from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class BoundaryTransformerConfig:
    vocab_size: int
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    @staticmethod
    def from_json(obj: Dict) -> "BoundaryTransformerConfig":
        return BoundaryTransformerConfig(**obj)

    def to_json(self) -> Dict:
        return asdict(self)


class BoundaryTransformer(nn.Module):
    """
    Character-level boundary predictor.

    Input: x [B, L] (ids)
    Output: logits [B, L-1] (boundary after each char)
    """

    def __init__(self, cfg: BoundaryTransformerConfig, pad_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id

        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(2048, cfg.d_model)  # large enough, clip in forward

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.drop = nn.Dropout(cfg.dropout)

        # use adjacent pair features
        self.proj = nn.Linear(cfg.d_model * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L]
        return logits: [B, L-1]
        """
        b, l = x.shape
        device = x.device

        # positional ids
        pos = torch.arange(l, device=device)
        pos = torch.clamp(pos, max=self.pos_emb.num_embeddings - 1)

        h = self.emb(x) + self.pos_emb(pos)[None, :, :]
        h = self.drop(h)

        # Transformer expects src_key_padding_mask: True for PAD positions
        pad_mask = x.eq(self.pad_id)
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        h_left = h[:, :-1, :]
        h_right = h[:, 1:, :]
        feat = torch.cat([h_left, h_right], dim=-1)
        logits = self.proj(self.drop(feat)).squeeze(-1)
        return logits
