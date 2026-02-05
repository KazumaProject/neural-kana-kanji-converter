from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from kkc.data import read_pairs_jsonl, split_train_valid, PairDataset, collate_fn
from kkc.model import TransformerConfig, Seq2SeqTransformer
from kkc.tokenizer import build_vocab, save_vocab
from kkc.utils import set_seed, get_device, save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=str, required=True, help="pairs.jsonl path")
    p.add_argument("--out_dir", type=str, required=True, help="output directory")
    p.add_argument("--valid_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_src_len", type=int, default=128)
    p.add_argument("--max_tgt_len", type=int, default=128)

    p.add_argument("--min_char_freq", type=int, default=1)

    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--enc_layers", type=int, default=4)
    p.add_argument("--dec_layers", type=int, default=4)
    p.add_argument("--ffn", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def lr_schedule(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    pairs_path = Path(args.pairs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = read_pairs_jsonl(pairs_path)
    if len(samples) < 10:
        raise RuntimeError(f"Too few samples: {len(samples)}")

    train_samples, valid_samples = split_train_valid(samples, valid_ratio=args.valid_ratio, seed=args.seed)

    # build vocab from train only (safe)
    src_vocab = build_vocab((s.src for s in train_samples), min_freq=args.min_char_freq)
    tgt_vocab = build_vocab((s.tgt for s in train_samples), min_freq=args.min_char_freq)

    save_vocab(src_vocab, out_dir / "src_vocab.json")
    save_vocab(tgt_vocab, out_dir / "tgt_vocab.json")

    train_ds = PairDataset(train_samples, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len)
    valid_ds = PairDataset(valid_samples, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
        pin_memory=(device.type == "cuda"),
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
        pin_memory=(device.type == "cuda"),
    )

    cfg = TransformerConfig(
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
        max_len=max(args.max_src_len, args.max_tgt_len) + 8,
    )

    model = Seq2SeqTransformer(cfg, pad_id=tgt_vocab.pad_id).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    best_valid = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"train epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            src = batch.src_ids.to(device)
            tgt_in = batch.tgt_in_ids.to(device)
            tgt_out = batch.tgt_out_ids.to(device)
            src_pad = batch.src_pad_mask.to(device)
            tgt_pad = batch.tgt_pad_mask.to(device)

            logits = model(
                src_ids=src,
                tgt_in_ids=tgt_in,
                src_key_padding_mask=src_pad,
                tgt_key_padding_mask=tgt_pad,
            )  # (B, T, V)

            # loss: ignore PAD
            V = logits.size(-1)
            loss = F.cross_entropy(
                logits.reshape(-1, V),
                tgt_out.reshape(-1),
                ignore_index=tgt_vocab.pad_id,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            lr = lr_schedule(global_step, args.lr, args.warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr

            opt.step()

            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), lr=lr)

        # validation
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in tqdm(valid_dl, desc=f"valid epoch {epoch}", dynamic_ncols=True):
                src = batch.src_ids.to(device)
                tgt_in = batch.tgt_in_ids.to(device)
                tgt_out = batch.tgt_out_ids.to(device)
                src_pad = batch.src_pad_mask.to(device)
                tgt_pad = batch.tgt_pad_mask.to(device)

                logits = model(
                    src_ids=src,
                    tgt_in_ids=tgt_in,
                    src_key_padding_mask=src_pad,
                    tgt_key_padding_mask=tgt_pad,
                )
                V = logits.size(-1)
                loss = F.cross_entropy(
                    logits.reshape(-1, V),
                    tgt_out.reshape(-1),
                    ignore_index=tgt_vocab.pad_id,
                    reduction="sum",
                )
                n = int(tgt_out.ne(tgt_vocab.pad_id).sum().item())
                total_loss += float(loss.item())
                total_tokens += n

        valid_nll = total_loss / max(1, total_tokens)
        valid_ppl = math.exp(min(20.0, valid_nll))
        print(f"[epoch {epoch}] valid_nll={valid_nll:.4f} valid_ppl={valid_ppl:.2f}")

        # save best
        if valid_nll < best_valid:
            best_valid = valid_nll
            ckpt = {
                "cfg": cfg.__dict__,
                "state_dict": model.state_dict(),
                "src_vocab_path": "src_vocab.json",
                "tgt_vocab_path": "tgt_vocab.json",
                "best_valid_nll": best_valid,
                "epoch": epoch,
            }
            save_checkpoint(out_dir / "model.pt", ckpt)
            print(f"saved: {out_dir / 'model.pt'}")

    print("done")


if __name__ == "__main__":
    main()
