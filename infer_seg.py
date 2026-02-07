from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from kkc.seg_model import BoundaryTransformer, BoundaryTransformerConfig
from kkc.tokenizer import load_vocab
from kkc.utils import get_device


# -------------------------
# Args
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="directory containing seg_model.pt, seg_config.json, vocab.json",
    )
    p.add_argument("--text", type=str, required=True, help="hiragana reading to segment")

    # 1-best mode (threshold)
    p.add_argument("--threshold", type=float, default=0.5, help="boundary probability threshold (used when --nbest=1)")

    # N-best mode (beam search)
    p.add_argument("--nbest", type=int, default=1, help="number of candidates to output (>=2 enables beam search)")
    p.add_argument("--beam_size", type=int, default=32, help="beam size for N-best decoding")
    p.add_argument(
        "--boundary_penalty",
        type=float,
        default=0.0,
        help="penalty per boundary to reduce over-segmentation (e.g. 0.05~0.3)",
    )

    # constraints
    p.add_argument("--min_seg_len", type=int, default=1, help="minimum segment length")
    p.add_argument("--max_seg_len", type=int, default=64, help="maximum segment length")

    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--json", action="store_true", help="output JSON instead of plain text")
    return p.parse_args()


# -------------------------
# Postprocess helpers (1-best threshold mode)
# -------------------------
def apply_min_max_len(s: str, boundaries: List[int], min_len: int, max_len: int) -> List[int]:
    """
    boundaries: list of split offsets in [1, len(s)-1]
    Post-process:
      - enforce max_len by inserting extra splits
      - enforce min_len by removing too-close splits (greedy)
    """
    n = len(s)
    if n <= 1:
        return []

    # 1) enforce max_len
    splits = sorted(set([b for b in boundaries if 1 <= b <= n - 1]))
    out: List[int] = []
    prev = 0
    for b in splits + [n]:
        while b - prev > max_len:
            prev2 = prev + max_len
            out.append(prev2)
            prev = prev2
        if b != n:
            out.append(b)
            prev = b
        else:
            prev = b

    # 2) enforce min_len (greedy)
    out2: List[int] = []
    prev = 0
    for b in out:
        if b - prev < min_len:
            continue
        out2.append(b)
        prev = b

    # also check tail segment
    if out2:
        last = out2[-1]
        if n - last < min_len:
            out2 = out2[:-1]

    return out2


def split_by_offsets(s: str, offsets: List[int]) -> List[str]:
    out: List[str] = []
    prev = 0
    for b in offsets:
        out.append(s[prev:b])
        prev = b
    out.append(s[prev:])
    return [x for x in out if x]


# -------------------------
# N-best decoding (beam search over boundaries)
# -------------------------
_EPS = 1e-9


def _log(x: float) -> float:
    return math.log(max(x, _EPS))


@dataclass
class BeamState:
    last_cut: int
    score: float
    boundaries: List[int]


def nbest_boundaries_beam(
    text: str,
    probs: List[float],  # len = n-1 (boundary after each char except last)
    nbest: int,
    beam_size: int,
    min_len: int,
    max_len: int,
    boundary_penalty: float,
) -> List[Tuple[List[int], float]]:
    """
    Find N-best boundary offset sequences.
    - boundary at offset k means split between text[k-1] and text[k]
    - offsets are in [1, n-1]

    Scoring:
      sum log(p_i) for cuts, log(1-p_i) for non-cuts, minus boundary_penalty per cut.

    Constraints enforced during search:
      - cannot cut if segment length < min_len
      - must cut if segment length >= max_len (hard cap)
      - tail segment (end) must be >= min_len
    """
    n = len(text)
    if n <= 1:
        return [([], 0.0)]

    beam: List[BeamState] = [BeamState(last_cut=0, score=0.0, boundaries=[])]

    for i, p in enumerate(probs):
        offset = i + 1  # cut after i-th char
        new_beam: List[BeamState] = []

        for st in beam:
            seg_len = offset - st.last_cut

            can_cut = (seg_len >= min_len)
            must_cut = (seg_len >= max_len)

            # option: no cut
            if not must_cut:
                new_beam.append(
                    BeamState(
                        last_cut=st.last_cut,
                        score=st.score + _log(1.0 - p),
                        boundaries=st.boundaries,
                    )
                )

            # option: cut
            if can_cut:
                new_beam.append(
                    BeamState(
                        last_cut=offset,
                        score=st.score + _log(p) - boundary_penalty,
                        boundaries=st.boundaries + [offset],
                    )
                )

        new_beam.sort(key=lambda s: s.score, reverse=True)
        beam = new_beam[: max(1, beam_size)]

    finals: List[Tuple[List[int], float]] = []
    for st in beam:
        tail_len = n - st.last_cut
        if tail_len < min_len:
            continue
        finals.append((st.boundaries, st.score))

    finals.sort(key=lambda x: x[1], reverse=True)

    # dedup by boundaries
    uniq: List[Tuple[List[int], float]] = []
    seen = set()
    for b, sc in finals:
        key = tuple(b)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((b, sc))
        if len(uniq) >= nbest:
            break

    return uniq


# -------------------------
# Main
# -------------------------
@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    model_dir = Path(args.model_dir)
    vocab = load_vocab(model_dir / "vocab.json")

    cfg_obj = json.loads((model_dir / "seg_config.json").read_text(encoding="utf-8"))
    cfg = BoundaryTransformerConfig.from_json(cfg_obj)

    ckpt = torch.load(str(model_dir / "seg_model.pt"), map_location="cpu")
    model = BoundaryTransformer(cfg, pad_id=vocab.pad_id)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    text = args.text.strip()
    if not text:
        if args.json:
            print(json.dumps({"reading_hira": "", "candidates": []}, ensure_ascii=False))
        else:
            print("")
        return

    x_ids = vocab.encode(text, add_bos=False, add_eos=False, max_len=None)
    if len(x_ids) <= 1:
        if args.json:
            print(json.dumps({"reading_hira": text, "candidates": [{"boundaries": [], "segments": [text], "score": 0.0}]}, ensure_ascii=False))
        else:
            print(text)
        return

    x = torch.tensor([x_ids], dtype=torch.long, device=device)
    logits = model(x)[0]  # [L-1]
    probs = torch.sigmoid(logits).detach().cpu().tolist()  # len = n-1

    # ---- 1-best (threshold) or N-best (beam) ----
    if args.nbest <= 1:
        boundaries = [i + 1 for i, p in enumerate(probs) if p >= args.threshold]
        boundaries = apply_min_max_len(text, boundaries, min_len=args.min_seg_len, max_len=args.max_seg_len)
        segments = split_by_offsets(text, boundaries)

        if args.json:
            out = {
                "reading_hira": text,
                "mode": "threshold",
                "threshold": args.threshold,
                "min_seg_len": args.min_seg_len,
                "max_seg_len": args.max_seg_len,
                "boundaries": boundaries,
                "segments": segments,
                "probs": probs,
            }
            print(json.dumps(out, ensure_ascii=False))
        else:
            print(" ".join(segments))
        return

    # N-best beam search
    cand = nbest_boundaries_beam(
        text=text,
        probs=probs,
        nbest=args.nbest,
        beam_size=args.beam_size,
        min_len=args.min_seg_len,
        max_len=args.max_seg_len,
        boundary_penalty=args.boundary_penalty,
    )

    if args.json:
        candidates = []
        for b, sc in cand:
            candidates.append(
                {
                    "boundaries": b,
                    "segments": split_by_offsets(text, b),
                    "score": sc,
                }
            )
        out = {
            "reading_hira": text,
            "mode": "beam",
            "nbest": args.nbest,
            "beam_size": args.beam_size,
            "boundary_penalty": args.boundary_penalty,
            "min_seg_len": args.min_seg_len,
            "max_seg_len": args.max_seg_len,
            "candidates": candidates,
            "probs": probs,
        }
        print(json.dumps(out, ensure_ascii=False))
    else:
        for rank, (b, sc) in enumerate(cand, 1):
            segs = split_by_offsets(text, b)
            print(f"{rank}: {' '.join(segs)}\t(score={sc:.3f})")


if __name__ == "__main__":
    main()
