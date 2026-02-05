from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .tokenizer import CharVocab
from .model import Seq2SeqTransformer


@dataclass
class Hyp:
    ids: List[int]
    logp: float
    ended: bool


@torch.no_grad()
def beam_search(
    model: Seq2SeqTransformer,
    src_ids: torch.Tensor,           # (1, S)
    src_pad_mask: torch.Tensor,      # (1, S)
    tgt_vocab: CharVocab,
    beam: int = 8,
    topk: int = 5,
    max_len: int = 128,
    len_norm_alpha: float = 0.6,
    device: torch.device | None = None,
) -> List[Tuple[str, float]]:
    """
    Returns: list of (decoded_text, score)
    score: normalized log-prob (higher is better)
    """
    if device is None:
        device = src_ids.device

    bos = tgt_vocab.bos_id
    eos = tgt_vocab.eos_id
    pad = tgt_vocab.pad_id

    # start with BOS
    hyps: List[Hyp] = [Hyp(ids=[bos], logp=0.0, ended=False)]

    for _step in range(max_len):
        # prepare batch of current hyps
        alive = [h for h in hyps if not h.ended]
        if not alive:
            break

        # (H, T)
        tgt_in = torch.tensor([h.ids for h in alive], dtype=torch.long, device=device)
        tgt_pad_mask = tgt_in.eq(pad)

        # expand src for each hyp
        src_rep = src_ids.expand(tgt_in.size(0), -1)
        src_mask_rep = src_pad_mask.expand(tgt_in.size(0), -1)

        logits = model(
            src_ids=src_rep,
            tgt_in_ids=tgt_in,
            src_key_padding_mask=src_mask_rep,
            tgt_key_padding_mask=tgt_pad_mask,
        )  # (H, T, V)

        next_logits = logits[:, -1, :]  # (H, V)
        next_logp = F.log_softmax(next_logits, dim=-1)  # (H, V)

        # collect candidates
        candidates: List[Hyp] = []

        for i, h in enumerate(alive):
            # pick top beam expansions per hyp
            lp, idx = torch.topk(next_logp[i], k=beam)
            for j in range(idx.size(0)):
                tok = int(idx[j].item())
                new_ids = h.ids + [tok]
                new_logp = h.logp + float(lp[j].item())
                ended = (tok == eos)
                candidates.append(Hyp(ids=new_ids, logp=new_logp, ended=ended))

        # keep best beams globally (including already ended hyps)
        ended_keep = [h for h in hyps if h.ended]
        merged = ended_keep + candidates

        def norm_score(h: Hyp) -> float:
            # length normalization on generated length excluding BOS
            gen_len = max(1, len(h.ids) - 1)
            denom = (gen_len ** len_norm_alpha)
            return h.logp / denom

        merged.sort(key=norm_score, reverse=True)
        hyps = merged[:beam]

        # early stop if enough ended
        if sum(1 for h in hyps if h.ended) >= topk:
            # still can continue, but usually enough
            pass

    # finalize: take best ended first; if none ended, take best alive
    def norm_score(h: Hyp) -> float:
        gen_len = max(1, len(h.ids) - 1)
        return h.logp / (gen_len ** len_norm_alpha)

    hyps.sort(key=norm_score, reverse=True)
    results: List[Tuple[str, float]] = []
    for h in hyps:
        # strip BOS and everything after EOS
        ids = h.ids[1:]
        if eos in ids:
            ids = ids[: ids.index(eos)]
        text = tgt_vocab.decode(ids, skip_special=True)
        if not text:
            continue
        results.append((text, norm_score(h)))
        if len(results) >= topk:
            break

    return results
