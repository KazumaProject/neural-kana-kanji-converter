from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


@dataclass(frozen=True)
class CharVocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def encode(self, s: str, add_bos: bool, add_eos: bool, max_len: int | None) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in s:
            ids.append(self.stoi.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out: List[str] = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            tok = self.itos[i]
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    def to_json(self) -> Dict:
        return {"itos": self.itos}

    @staticmethod
    def from_json(obj: Dict) -> "CharVocab":
        itos = list(obj["itos"])
        stoi = {t: i for i, t in enumerate(itos)}
        # safety: ensure specials exist
        for t in SPECIAL_TOKENS:
            if t not in stoi:
                itos.insert(0, t)
        stoi = {t: i for i, t in enumerate(itos)}
        return CharVocab(stoi=stoi, itos=itos)


def build_vocab(texts: Iterable[str], min_freq: int = 1) -> CharVocab:
    freq: Dict[str, int] = {}
    for s in texts:
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1

    chars = [ch for ch, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])) if c >= min_freq]
    itos = SPECIAL_TOKENS + chars
    stoi = {t: i for i, t in enumerate(itos)}
    return CharVocab(stoi=stoi, itos=itos)


def save_vocab(vocab: CharVocab, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab(path: Path) -> CharVocab:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return CharVocab.from_json(obj)
