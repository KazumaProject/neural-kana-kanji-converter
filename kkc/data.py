from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import CharVocab


@dataclass(frozen=True)
class PairSample:
    src: str  # reading_hira
    tgt: str  # surface


def read_pairs_jsonl(path: Path) -> List[PairSample]:
    samples: List[PairSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            src = str(obj.get("reading_hira", ""))
            tgt = str(obj.get("surface", ""))
            if not src or not tgt:
                continue
            samples.append(PairSample(src=src, tgt=tgt))
    return samples


def split_train_valid(samples: List[PairSample], valid_ratio: float, seed: int) -> Tuple[List[PairSample], List[PairSample]]:
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_valid = int(len(samples) * valid_ratio)
    valid = [samples[i] for i in idx[:n_valid]]
    train = [samples[i] for i in idx[n_valid:]]
    return train, valid


def pad_1d(ids: List[int], pad_id: int, max_len: int) -> List[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


@dataclass(frozen=True)
class Batch:
    src_ids: torch.Tensor  # (B, S)
    tgt_in_ids: torch.Tensor  # (B, T)  teacher forcing input (BOS + ...)
    tgt_out_ids: torch.Tensor  # (B, T) expected output (... + EOS)
    src_pad_mask: torch.Tensor  # (B, S) True where PAD
    tgt_pad_mask: torch.Tensor  # (B, T) True where PAD


class PairDataset(Dataset):
    def __init__(
        self,
        samples: List[PairSample],
        src_vocab: CharVocab,
        tgt_vocab: CharVocab,
        max_src_len: int,
        max_tgt_len: int,
    ) -> None:
        self.samples = samples
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[List[int], List[int], List[int]]:
        s = self.samples[i]
        # src: BOS .. EOS
        src_ids = self.src_vocab.encode(s.src, add_bos=True, add_eos=True, max_len=self.max_src_len)
        # tgt: prepare teacher forcing
        tgt_full = self.tgt_vocab.encode(s.tgt, add_bos=True, add_eos=True, max_len=self.max_tgt_len)
        # input excludes last, output excludes first
        tgt_in = tgt_full[:-1]
        tgt_out = tgt_full[1:]
        return src_ids, tgt_in, tgt_out


def collate_fn(
    batch: List[Tuple[List[int], List[int], List[int]]],
    src_pad_id: int,
    tgt_pad_id: int,
) -> Batch:
    src_max = max(len(x[0]) for x in batch)
    tgt_max = max(len(x[1]) for x in batch)

    src_ids = torch.tensor([pad_1d(x[0], src_pad_id, src_max) for x in batch], dtype=torch.long)
    tgt_in = torch.tensor([pad_1d(x[1], tgt_pad_id, tgt_max) for x in batch], dtype=torch.long)
    tgt_out = torch.tensor([pad_1d(x[2], tgt_pad_id, tgt_max) for x in batch], dtype=torch.long)

    src_pad_mask = src_ids.eq(src_pad_id)  # True at PAD
    tgt_pad_mask = tgt_in.eq(tgt_pad_id)

    return Batch(
        src_ids=src_ids,
        tgt_in_ids=tgt_in,
        tgt_out_ids=tgt_out,
        src_pad_mask=src_pad_mask,
        tgt_pad_mask=tgt_pad_mask,
    )
