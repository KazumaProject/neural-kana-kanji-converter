from __future__ import annotations

"""Rerank a *given* candidate list (model-agnostic).

This script does NOT depend on the generator model.

You provide:
  - input reading (and optional left/right context)
  - a list of conversion candidates (surfaces)

Supported candidate inputs:
  1) Repeated --candidate arguments
  2) --candidates "cand1|cand2|..." with --delimiter
  3) --candidates_json FILE
     - JSON array of strings: ["候補1", "候補2", ...]
     - or JSON array of objects: [{"candidate":"候補1","base_score":-1.2}, ...]

If base_score is provided, you can blend it:
  final = rerank + alpha_base * base_score

Examples:
  python infer_rerank_candidates.py \
    --rerank_model_dir out_rerank \
    --text わたしのなまえはなかのです \
    --candidate 私の名前は中野です --candidate わたしの名前は中野です

  python infer_rerank_candidates.py \
    --rerank_model_dir out_rerank \
    --left アルツハイマー病 --text におけるやくわり --right を調べた \
    --candidates "における役割|における訳割" \
    --alpha_base 0.2
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from kkc.data import pack_src_with_context
from kkc.rerank_model import RerankTransformer, RerankTransformerConfig
from kkc.tokenizer import CharVocab, load_vocab
from kkc.utils import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rerank_model_dir",
        type=str,
        required=True,
        help="directory containing rerank_model.pt and rerank_vocab.json",
    )

    # Input (same interface as infer.py / infer_rerank.py)
    p.add_argument("--text", type=str, required=True, help="hiragana reading to convert (conversion segment)")
    p.add_argument("--left", type=str, default="", help="left context (already committed text; optional)")
    p.add_argument("--right", type=str, default="", help="right context (text after cursor; optional)")

    # Candidates
    p.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="conversion candidate surface (repeatable)",
    )
    p.add_argument(
        "--candidates",
        type=str,
        default="",
        help="single string that contains multiple candidates separated by --delimiter",
    )
    p.add_argument(
        "--delimiter",
        type=str,
        default="|",
        help="delimiter for --candidates (default: '|')",
    )
    p.add_argument(
        "--candidates_json",
        type=str,
        default="",
        help="path to JSON file containing candidates (array of strings or objects)",
    )

    p.add_argument("--alpha_base", type=float, default=0.0, help="blend base score: final = rerank + alpha_base * base")
    p.add_argument("--max_len", type=int, default=256, help="max sequence length for reranker input")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    return p.parse_args()


def _load_reranker(model_dir: Path, device: torch.device) -> Tuple[RerankTransformer, CharVocab, str]:
    ckpt = torch.load(str(model_dir / "rerank_model.pt"), map_location="cpu")
    cfg = RerankTransformerConfig(**ckpt["cfg"])
    vocab = load_vocab(model_dir / ckpt["vocab_path"])
    sep_char = str(ckpt.get("sep_char", "␞"))

    model = RerankTransformer(cfg, pad_id=vocab.pad_id)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, sep_char


def _encode(vocab: CharVocab, src_text: str, cand: str, sep_char: str, max_len: int) -> List[int]:
    text = f"{src_text}{sep_char}{cand}"
    return vocab.encode(text, add_bos=True, add_eos=True, max_len=max_len)


def _pad_2d(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    maxlen = max(len(x) for x in seqs)
    out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    for i, x in enumerate(seqs):
        out[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return out


def _load_candidates(args: argparse.Namespace) -> List[Tuple[str, Optional[float]]]:
    """Returns list of (candidate_surface, base_score_or_None)."""
    out: List[Tuple[str, Optional[float]]] = []

    # 1) JSON file
    if args.candidates_json:
        path = Path(args.candidates_json)
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        out.append((s, None))
                elif isinstance(item, dict):
                    cand = str(item.get("candidate", item.get("cand", item.get("surface", "")))).strip()
                    if not cand:
                        continue
                    base = item.get("base_score", item.get("score", None))
                    base_f: Optional[float]
                    try:
                        base_f = float(base) if base is not None else None
                    except Exception:
                        base_f = None
                    out.append((cand, base_f))
                else:
                    # ignore unknown
                    continue
        else:
            raise ValueError("--candidates_json must be a JSON array")

    # 2) Delimited candidates string
    if args.candidates:
        parts = [x.strip() for x in args.candidates.split(args.delimiter)]
        for s in parts:
            if s:
                out.append((s, None))

    # 3) Repeated --candidate
    for s in (args.candidate or []):
        s = str(s).strip()
        if s:
            out.append((s, None))

    # Dedup while preserving order
    seen = set()
    uniq: List[Tuple[str, Optional[float]]] = []
    for cand, base in out:
        if cand in seen:
            continue
        seen.add(cand)
        uniq.append((cand, base))
    return uniq


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    rr_model, rr_vocab, sep_char = _load_reranker(Path(args.rerank_model_dir), device)

    src_text = args.text
    if args.left or args.right:
        src_text = pack_src_with_context(left=args.left, reading_hira=args.text, right=args.right)

    cands = _load_candidates(args)
    if not cands:
        raise SystemExit("No candidates provided. Use --candidate, --candidates, or --candidates_json")

    cand_texts = [c for c, _ in cands]
    base_scores = [b for _, b in cands]

    ids = [_encode(rr_vocab, src_text, c, sep_char, max_len=min(args.max_len, rr_model.cfg.max_len)) for c in cand_texts]
    x = _pad_2d(ids, rr_vocab.pad_id).to(device)
    pad_mask = x.eq(rr_vocab.pad_id)

    with torch.no_grad():
        rr_scores = rr_model(x, pad_mask).tolist()

    final: List[Tuple[str, Optional[float], float, float]] = []
    for cand, base, r in zip(cand_texts, base_scores, rr_scores):
        b = float(base) if base is not None else 0.0
        f = float(r) + args.alpha_base * b
        final.append((cand, base, float(r), float(f)))

    final.sort(key=lambda t: t[3], reverse=True)

    for i, (cand, base, r, f) in enumerate(final, start=1):
        if base is None:
            print(f"{i}\tfinal={f:.4f}\trerank={r:.4f}\t{cand}")
        else:
            print(f"{i}\tfinal={f:.4f}\trerank={r:.4f}\tbase={float(base):.4f}\t{cand}")


if __name__ == "__main__":
    main()
