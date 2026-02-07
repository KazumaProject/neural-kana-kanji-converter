from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from kkc.seg_data import read_seg_jsonl, SegDataset, seg_collate_fn, SegSample
from kkc.seg_model import BoundaryTransformer, BoundaryTransformerConfig
from kkc.tokenizer import build_vocab, save_vocab
from kkc.utils import set_seed, get_device, save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # CHANGED: allow multiple
    p.add_argument(
        "--pairs_seg",
        type=str,
        required=True,
        nargs="+",
        help="one or more seg jsonl paths made by make_pairs_unified.py --task seg_boundary",
    )
    p.add_argument("--out_dir", type=str, required=True, help="output directory")
    p.add_argument("--valid_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--min_char_freq", type=int, default=1)

    # model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # train
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])

    # OPTIONAL: dataset hygiene
    p.add_argument(
        "--dedup",
        action="store_true",
        help="deduplicate samples across files by reading_hira + y pattern (recommended when mixing corpora)",
    )
    return p.parse_args()


def split_train_valid(samples: List[SegSample], valid_ratio: float, seed: int) -> Tuple[List[SegSample], List[SegSample]]:
    if valid_ratio <= 0.0:
        return samples, []
    import random

    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_valid = max(1, int(len(samples) * valid_ratio))
    valid_idx = set(idx[:n_valid])
    train, valid = [], []
    for i, s in enumerate(samples):
        (valid if i in valid_idx else train).append(s)
    return train, valid


def bce_loss_masked(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, L-1]
    y: [B, L-1] float, padded with -1
    mask: [B, L-1] bool True for valid positions
    """
    y_valid = torch.where(mask, y, torch.zeros_like(y))
    loss = F.binary_cross_entropy_with_logits(logits, y_valid, reduction="none")
    loss = torch.where(mask, loss, torch.zeros_like(loss))
    denom = mask.sum().clamp_min(1)
    return loss.sum() / denom


@torch.no_grad()
def f1_from_logits(
    logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, thr: float = 0.5
) -> Tuple[float, float, float]:
    probs = torch.sigmoid(logits)
    pred = probs.ge(thr)
    gold = y.ge(0.5)

    pred = pred & mask
    gold = gold & mask

    tp = (pred & gold).sum().item()
    fp = (pred & ~gold).sum().item()
    fn = (~pred & gold).sum().item()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _load_all_seg_samples(paths: List[str], dedup: bool) -> List[SegSample]:
    all_samples: List[SegSample] = []
    for s in paths:
        p = Path(s)
        if not p.exists():
            raise FileNotFoundError(f"--pairs_seg file not found: {p}")
        if p.is_dir():
            raise IsADirectoryError(f"--pairs_seg must be files, got directory: {p}")
        samples = read_seg_jsonl(p)
        all_samples.extend(samples)

    if not dedup:
        return all_samples

    # Dedup by (reading_hira, y) — assumes SegSample has at least these attributes.
    # If your SegSample differs, adjust the key accordingly.
    seen = set()
    out: List[SegSample] = []
    for s in all_samples:
        key = (getattr(s, "reading_hira", None), getattr(s, "y", None))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CHANGED: load multiple files
    samples = _load_all_seg_samples(args.pairs_seg, dedup=args.dedup)
    if len(samples) == 0:
        raise ValueError("No samples loaded. Check --pairs_seg inputs.")

    train_samples, valid_samples = split_train_valid(samples, args.valid_ratio, args.seed)

    # vocab from train split only
    vocab = build_vocab((s.reading_hira for s in train_samples), min_freq=args.min_char_freq)

    # FIX: correct argument order (vocab first, path second)
    save_vocab(vocab, out_dir / "vocab.json")

    cfg = BoundaryTransformerConfig(
        vocab_size=len(vocab.itos),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    (out_dir / "seg_config.json").write_text(json.dumps(cfg.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    model = BoundaryTransformer(cfg, pad_id=vocab.pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds = SegDataset(train_samples, vocab=vocab, max_len=args.max_len)
    valid_ds = SegDataset(valid_samples, vocab=vocab, max_len=args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: seg_collate_fn(b, pad_id=vocab.pad_id),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: seg_collate_fn(b, pad_id=vocab.pad_id),
    )

    best_f1 = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        for batch in pbar:
            x = batch.x.to(device)
            y = batch.y.to(device)

            logits = model(x)  # [B, L-1]

            pad_mask = batch.x_pad_mask.to(device)  # [B, L]
            gap_mask = (~pad_mask[:, :-1]) & (~pad_mask[:, 1:])
            gap_mask = gap_mask & y.ne(-1.0)

            loss = bce_loss_masked(logits, y, gap_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            global_step += 1
            if global_step % 50 == 0:
                prec, rec, f1 = f1_from_logits(logits.detach(), y, gap_mask)
                pbar.set_postfix(loss=float(loss.item()), f1=f1)

        # validation
        model.eval()
        if len(valid_ds) > 0:
            v_loss_sum = 0.0
            v_count = 0
            tp = fp = fn = 0
            for batch in tqdm(valid_loader, desc=f"valid epoch {epoch}", leave=False):
                x = batch.x.to(device)
                y = batch.y.to(device)
                logits = model(x)

                pad_mask = batch.x_pad_mask.to(device)
                gap_mask = (~pad_mask[:, :-1]) & (~pad_mask[:, 1:])
                gap_mask = gap_mask & y.ne(-1.0)

                loss = bce_loss_masked(logits, y, gap_mask)
                v_loss_sum += float(loss.item())
                v_count += 1

                probs = torch.sigmoid(logits)
                pred = probs.ge(0.5) & gap_mask
                gold = y.ge(0.5) & gap_mask
                tp += (pred & gold).sum().item()
                fp += (pred & ~gold).sum().item()
                fn += (~pred & gold).sum().item()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            v_loss = v_loss_sum / max(v_count, 1)

            print(f"[epoch {epoch}] valid_loss={v_loss:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                save_checkpoint(out_dir / "seg_model.pt", {"model": model.state_dict(), "cfg": cfg.to_json()})
                print(f"  saved best to {out_dir/'seg_model.pt'}")
        else:
            save_checkpoint(out_dir / "seg_model.pt", {"model": model.state_dict(), "cfg": cfg.to_json()})
            print(f"[epoch {epoch}] saved to {out_dir/'seg_model.pt'}")

    print("done")


if __name__ == "__main__":
    main()
