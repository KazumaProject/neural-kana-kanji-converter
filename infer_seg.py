from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch

from kkc.seg_model import BoundaryTransformer, BoundaryTransformerConfig
from kkc.tokenizer import load_vocab
from kkc.utils import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True, help="directory containing seg_model.pt, seg_config.json, vocab.json")
    p.add_argument("--text", type=str, required=True, help="hiragana reading to segment")
    p.add_argument("--threshold", type=float, default=0.5, help="boundary probability threshold")
    p.add_argument("--min_seg_len", type=int, default=1, help="minimum segment length (post-process)")
    p.add_argument("--max_seg_len", type=int, default=64, help="maximum segment length (post-process)")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--json", action="store_true", help="output JSON instead of plain text")
    return p.parse_args()


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
            # drop this boundary
            continue
        out2.append(b)
        prev = b

    # also check tail segment
    if out2:
        last = out2[-1]
        if n - last < min_len:
            # drop last boundary
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


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    model_dir = Path(args.model_dir)
    vocab = load_vocab(model_dir / "vocab.json")

    # load cfg and weights
    cfg_obj = json.loads((model_dir / "seg_config.json").read_text(encoding="utf-8"))
    cfg = BoundaryTransformerConfig.from_json(cfg_obj)

    ckpt = torch.load(str(model_dir / "seg_model.pt"), map_location="cpu")
    model = BoundaryTransformer(cfg, pad_id=vocab.pad_id)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    text = args.text.strip()
    if not text:
        print("")
        return

    x_ids = vocab.encode(text, add_bos=False, add_eos=False, max_len=None)
    if len(x_ids) <= 1:
        if args.json:
            print(json.dumps({"reading_hira": text, "boundaries": [], "segments": [text]}, ensure_ascii=False))
        else:
            print(text)
        return

    x = torch.tensor([x_ids], dtype=torch.long, device=device)
    logits = model(x)[0]  # [L-1]
    probs = torch.sigmoid(logits).detach().cpu().tolist()

    boundaries = [i + 1 for i, p in enumerate(probs) if p >= args.threshold]  # offset after i-th char
    boundaries = apply_min_max_len(text, boundaries, min_len=args.min_seg_len, max_len=args.max_seg_len)
    segments = split_by_offsets(text, boundaries)

    if args.json:
        print(json.dumps({"reading_hira": text, "boundaries": boundaries, "segments": segments, "probs": probs}, ensure_ascii=False))
    else:
        print(" ".join(segments))


if __name__ == "__main__":
    main()
