from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kkc.model import TransformerConfig, Seq2SeqTransformer
from kkc.tokenizer import load_vocab
from kkc.decode import beam_search
from kkc.utils import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True, help="directory containing model.pt and vocabs")
    p.add_argument("--text", type=str, required=True, help="hiragana reading input")
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    model_dir = Path(args.model_dir)
    ckpt = torch.load(str(model_dir / "model.pt"), map_location="cpu")

    src_vocab = load_vocab(model_dir / ckpt["src_vocab_path"])
    tgt_vocab = load_vocab(model_dir / ckpt["tgt_vocab_path"])

    cfg = TransformerConfig(**ckpt["cfg"])
    model = Seq2SeqTransformer(cfg, pad_id=tgt_vocab.pad_id)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    src_ids = src_vocab.encode(args.text, add_bos=True, add_eos=True, max_len=cfg.max_len)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad_mask = src.eq(src_vocab.pad_id)

    results = beam_search(
        model=model,
        src_ids=src,
        src_pad_mask=src_pad_mask,
        tgt_vocab=tgt_vocab,
        beam=args.beam,
        topk=args.topk,
        max_len=args.max_len,
        device=device,
    )

    for i, (cand, score) in enumerate(results, start=1):
        print(f"{i}\t{score:.4f}\t{cand}")


if __name__ == "__main__":
    main()
