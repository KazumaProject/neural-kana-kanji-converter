from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import CharVocab


@dataclass(frozen=True)
class SegSample:
    """
    Boundary segmentation sample.

    reading_hira: hiragana string
    boundary: string of length len(reading_hira)-1, each char is "0" or "1"
              boundary[i] == "1" means there is a split AFTER reading_hira[i]
    """
    reading_hira: str
    boundary: str


def read_seg_jsonl(path: Path) -> List[SegSample]:
    out: List[SegSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            reading = str(obj.get("reading_hira", ""))
            boundary = str(obj.get("boundary", ""))
            if not reading:
                continue
            if len(reading) <= 1:
                continue
            if len(boundary) != len(reading) - 1:
                continue
            if any(c not in ("0", "1") for c in boundary):
                continue
            out.append(SegSample(reading_hira=reading, boundary=boundary))
    return out


def pad_1d(x: List[int], pad: int, max_len: int) -> List[int]:
    if len(x) >= max_len:
        return x[:max_len]
    return x + [pad] * (max_len - len(x))


def pad_1d_float(x: List[float], pad: float, max_len: int) -> List[float]:
    if len(x) >= max_len:
        return x[:max_len]
    return x + [pad] * (max_len - len(x))


@dataclass
class SegBatch:
    x: torch.Tensor         # [B, L] long
    y: torch.Tensor         # [B, L-1] float, padded with -1
    x_pad_mask: torch.Tensor  # [B, L] bool (True for PAD positions)


class SegDataset(Dataset):
    def __init__(self, samples: List[SegSample], vocab: CharVocab, max_len: int) -> None:
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[float]]:
        s = self.samples[idx]
        # No BOS/EOS for boundary task
        x_ids = self.vocab.encode(s.reading_hira, add_bos=False, add_eos=False, max_len=self.max_len)
        # boundary needs to align with characters; if truncated, truncate boundary too.
        # boundary length should be len(chars)-1.
        n = len(x_ids)
        if n <= 1:
            # will be filtered out by collate via mask
            y = []
        else:
            y = [1.0 if c == "1" else 0.0 for c in s.boundary[: n - 1]]
        return x_ids, y


def seg_collate_fn(batch: List[Tuple[List[int], List[float]]], pad_id: int) -> SegBatch:
    # NOTE: boundary y is length len(x)-1
    max_x = max(len(x) for x, _ in batch) if batch else 1
    max_x = max(max_x, 2)

    x_pad = torch.tensor([pad_1d(x, pad_id, max_x) for x, _ in batch], dtype=torch.long)
    pad_mask = x_pad.eq(pad_id)  # True where pad

    # y padding: max_x-1
    max_y = max_x - 1
    y_pad = torch.tensor([pad_1d_float(y, -1.0, max_y) for _, y in batch], dtype=torch.float32)

    return SegBatch(x=x_pad, y=y_pad, x_pad_mask=pad_mask)
