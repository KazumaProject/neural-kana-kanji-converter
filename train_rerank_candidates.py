from __future__ import annotations

"""Train a reranker from *external* candidate lists (model-agnostic).

This script does NOT depend on the generator model.

Training data is JSONL. Each line must include:

  - input: either
      (A) {"src": "..."}
          (already packed, e.g. left⟨reading⟩right)
      or
      (B) {"reading_hira": "...", "left": "...", "right": "..."}
          (src is built as left⟨reading_hira⟩right; left/right optional)

  - gold (correct surface): one of
      - "gold"
      - "surface"
      - "tgt"

  - candidates: an ordered list of candidate surfaces
      - "candidates": ["cand1", "cand2", ...]

Notes:
  * Candidates are assumed to be in the original system's priority order (best first).
  * We typically train on top K candidates (K=8..16).
  * If the gold surface is not present within top K, the sample is skipped by default
    (because a reranker cannot pick an answer that is not in the list).

Outputs (in --out_dir):
  - rerank_vocab.json
  - rerank_model.pt

Example:
  python train_rerank_candidates.py --data rerank_train.jsonl --out_dir out_rerank --topk 16 --epochs 3 --device cuda
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from kkc.data import pack_src_with_context
from kkc.rerank_model import RerankTransformer, RerankTransformerConfig
from kkc.tokenizer import CharVocab, build_vocab, load_vocab, save_vocab
from kkc.utils import get_device, save_checkpoint, set_seed


# A single character separator between (src) and (candidate surface).
# Choose a rarely-used control picture to avoid collisions.
SEP_CHAR = "␞"  # U+241E SYMBOL FOR RECORD SEPARATOR


@dataclass(frozen=True)
class RerankCandSample:
    src: str
    gold: str
    candidates: List[str]  # ordered (best first)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data", type=str, nargs="+", required=True, help="rerank candidates JSONL path(s)")
    p.add_argument("--out_dir", type=str, required=True, help="output directory (rerank_model.pt, rerank_vocab.json)")
    p.add_argument("--valid_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)

    # candidates
    p.add_argument("--topk", type=int, default=16, help="use top K candidates from each list")
    p.add_argument("--min_candidates", type=int, default=2, help="skip examples with fewer than this many candidates")

    # vocab/model
    p.add_argument("--min_char_freq", type=int, default=1)
    p.add_argument("--max_seq_len", type=int, default=256, help="max input length for reranker (src+sep+cand)")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ffn", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # optimization
    p.add_argument("--batch_size", type=int, default=8, help="number of groups per optimizer step")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # behaviour
    p.add_argument(
        "--inject_gold_if_missing",
        action="store_true",
        help="if gold is not in topK candidates, append it (weakly assumes gold is valid candidate)",
    )

    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    return p.parse_args()


def lr_schedule(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _coalesce_str(obj: dict, keys: List[str]) -> str:
    for k in keys:
        v = obj.get(k, None)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _load_samples(paths: List[Path]) -> List[RerankCandSample]:
    out: List[RerankCandSample] = []
    for p in paths:
        if not p.exists():
            raise RuntimeError(f"data file not found: {p}")

        for obj in _read_jsonl(p):
            # src
            src = _coalesce_str(obj, ["src"])
            if not src:
                reading = _coalesce_str(obj, ["reading_hira", "reading", "text"])  # allow aliases
                left = _coalesce_str(obj, ["left", "left_context"])
                right = _coalesce_str(obj, ["right", "right_context"])
                if reading:
                    if left or right:
                        src = pack_src_with_context(left=left, reading_hira=reading, right=right)
                    else:
                        src = reading

            gold = _coalesce_str(obj, ["gold", "surface", "tgt", "target", "answer"])

            cands = obj.get("candidates", obj.get("nbest", None))
            if not isinstance(cands, list):
                continue

            cand_list: List[str] = []
            for c in cands:
                if c is None:
                    continue
                s = str(c).strip()
                if s:
                    cand_list.append(s)

            # de-dup candidates while preserving order
            seen = set()
            uniq: List[str] = []
            for c in cand_list:
                if c in seen:
                    continue
                seen.add(c)
                uniq.append(c)

            if not src or not gold or len(uniq) == 0:
                continue

            out.append(RerankCandSample(src=src, gold=gold, candidates=uniq))

    return out


def _split_train_valid(samples: List[RerankCandSample], valid_ratio: float, seed: int) -> Tuple[List[RerankCandSample], List[RerankCandSample]]:
    import random

    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_valid = int(len(samples) * valid_ratio)
    valid = [samples[i] for i in idx[:n_valid]]
    train = [samples[i] for i in idx[n_valid:]]
    return train, valid


def _encode(vocab: CharVocab, src: str, cand: str, max_len: int) -> List[int]:
    text = f"{src}{SEP_CHAR}{cand}"
    return vocab.encode(text, add_bos=True, add_eos=True, max_len=max_len)


def _pad_2d(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    maxlen = max(len(x) for x in seqs)
    out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    for i, x in enumerate(seqs):
        out[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(p) for p in args.data]
    samples = _load_samples(paths)
    if not samples:
        raise RuntimeError("no samples loaded (check --data and JSONL schema)")

    train_s, valid_s = _split_train_valid(samples, valid_ratio=args.valid_ratio, seed=args.seed)

    # Build vocab from train split only (avoid peeking at valid).
    texts: List[str] = []
    for s in train_s:
        texts.append(s.src)
        texts.append(s.gold)
        for c in s.candidates[: args.topk]:
            texts.append(c)
    texts.append(SEP_CHAR)

    rr_vocab = build_vocab(texts, min_freq=args.min_char_freq)
    vocab_path = out_dir / "rerank_vocab.json"
    save_vocab(rr_vocab, vocab_path)

    rr_cfg = RerankTransformerConfig(
        vocab_size=len(rr_vocab.itos),
        max_len=args.max_seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        ffn=args.ffn,
        dropout=args.dropout,
    )
    rr_model = RerankTransformer(rr_cfg, pad_id=rr_vocab.pad_id).to(device)

    opt = torch.optim.AdamW(rr_model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    step = 0

    def run_eval(split: List[RerankCandSample], desc: str) -> None:
        nonlocal step
        rr_model.eval()

        used = 0
        skipped = 0
        top1_ok = 0
        oracle = 0

        for s in tqdm(split, desc=desc, leave=False):
            cand_texts = s.candidates[: args.topk]
            if len(cand_texts) < args.min_candidates:
                skipped += 1
                continue

            if s.gold in cand_texts:
                oracle += 1
                y = cand_texts.index(s.gold)
            else:
                if args.inject_gold_if_missing:
                    cand_texts = cand_texts + [s.gold]
                    y = len(cand_texts) - 1
                else:
                    skipped += 1
                    continue

            ids = [_encode(rr_vocab, s.src, c, args.max_seq_len) for c in cand_texts]
            x = _pad_2d(ids, rr_vocab.pad_id).to(device)
            pad_mask = x.eq(rr_vocab.pad_id)

            scores = rr_model(x, pad_mask)
            pred = int(torch.argmax(scores).item())

            used += 1
            if pred == y:
                top1_ok += 1

        rr_model.train()
        oracle_rate = oracle / max(1, len(split))
        acc = top1_ok / max(1, used)
        print(f"[{desc}] used={used} skipped={skipped} oracle@{args.topk}={oracle_rate:.4f} top1_acc={acc:.4f}")

    rr_model.train()
    print(f"train={len(train_s)} valid={len(valid_s)} topk={args.topk} SEP_CHAR={SEP_CHAR} vocab={len(rr_vocab.itos)}")

    for epoch in range(1, args.epochs + 1):
        rr_model.train()
        loss_sum = 0.0
        groups = 0
        skipped = 0
        oracle = 0

        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_s, desc=f"epoch {epoch}/{args.epochs}")

        for s in pbar:
            cand_texts = s.candidates[: args.topk]
            if len(cand_texts) < args.min_candidates:
                skipped += 1
                continue

            if s.gold in cand_texts:
                oracle += 1
                y = cand_texts.index(s.gold)
            else:
                if args.inject_gold_if_missing:
                    cand_texts = cand_texts + [s.gold]
                    y = len(cand_texts) - 1
                else:
                    skipped += 1
                    continue

            ids = [_encode(rr_vocab, s.src, c, args.max_seq_len) for c in cand_texts]
            x = _pad_2d(ids, rr_vocab.pad_id).to(device)
            pad_mask = x.eq(rr_vocab.pad_id)

            scores = rr_model(x, pad_mask).unsqueeze(0)  # (1, N)
            target = torch.tensor([y], dtype=torch.long, device=device)

            loss = F.cross_entropy(scores, target)
            loss.backward()

            loss_sum += float(loss.item())
            groups += 1

            if (groups % args.batch_size) == 0:
                step_lr = lr_schedule(step, args.lr, args.warmup_steps)
                for pg in opt.param_groups:
                    pg["lr"] = step_lr

                torch.nn.utils.clip_grad_norm_(rr_model.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

            pbar.set_postfix(loss=loss_sum / max(1, groups), oracle=oracle / max(1, groups + skipped), skipped=skipped)

        # final partial batch update
        if groups % args.batch_size != 0:
            step_lr = lr_schedule(step, args.lr, args.warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = step_lr
            torch.nn.utils.clip_grad_norm_(rr_model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

        print(f"[train] epoch={epoch} groups_used={groups} skipped={skipped} oracle@{args.topk}={(oracle / max(1, groups + skipped)):.4f} loss={loss_sum / max(1, groups):.4f}")
        run_eval(valid_s, desc=f"valid epoch {epoch}")

        ckpt = {
            "type": "reranker",
            "cfg": rr_cfg.__dict__,
            "state_dict": rr_model.state_dict(),
            "vocab_path": "rerank_vocab.json",
            "sep_char": SEP_CHAR,
            "epoch": epoch,
            "step": step,
            "topk": args.topk,
            "data": [str(p) for p in paths],
        }
        save_checkpoint(out_dir / "rerank_model.pt", ckpt)

    print(f"saved: {out_dir / 'rerank_model.pt'}")


if __name__ == "__main__":
    main()
