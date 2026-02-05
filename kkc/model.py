from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 512  # for positional encoding


class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig, pad_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id

        self.src_embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=pad_id)

        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.dropout, max_len=cfg.max_len)

        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )

        self.lm_head = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        # True where masked (upper triangle)
        return torch.triu(torch.ones((sz, sz), device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        src_ids: torch.Tensor,         # (B, S)
        tgt_in_ids: torch.Tensor,      # (B, T)
        src_key_padding_mask: torch.Tensor,  # (B, S) True at PAD
        tgt_key_padding_mask: torch.Tensor,  # (B, T) True at PAD
    ) -> torch.Tensor:
        device = src_ids.device
        B, S = src_ids.shape
        _, T = tgt_in_ids.shape

        src = self.src_embed(src_ids) * math.sqrt(self.cfg.d_model)
        tgt = self.tgt_embed(tgt_in_ids) * math.sqrt(self.cfg.d_model)

        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt)

        tgt_mask = self._generate_square_subsequent_mask(T, device=device)

        out = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (B, T, D)

        logits = self.lm_head(out)  # (B, T, V)
        return logits
