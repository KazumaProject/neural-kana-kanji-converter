from __future__ import annotations

import argparse
import itertools
import math
import random
from pathlib import Path
from typing import List, Set, Tuple

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

    # -------------------------
    # Pairs input
    # -------------------------
    g = p.add_argument_group("pairs")
    g.add_argument(
        "--pairs",
        type=str,
        nargs="+",
        default=None,
        help="pairs.jsonl paths (one or more). If provided, mixes by simple merge (no probabilistic mixing).",
    )
    g.add_argument(
        "--pairs_sentence",
        type=str,
        nargs="+",
        default=None,
        help="sentence-mode pairs.jsonl paths (one or more). Used with probabilistic mixing.",
    )
    g.add_argument(
        "--pairs_span",
        type=str,
        nargs="+",
        default=None,
        help="span-mode pairs.jsonl paths (one or more). Used with probabilistic mixing.",
    )

    # Mixing schedule
    mg = p.add_argument_group("mixing")
    mg.add_argument(
        "--mix_schedule",
        type=str,
        default="fixed",
        choices=["fixed", "linear", "cosine"],
        help="how to schedule span probability across training.",
    )
    mg.add_argument(
        "--mix_span_prob",
        type=float,
        default=0.5,
        help="fixed probability to draw from span dataset (used when --mix_schedule=fixed).",
    )
    mg.add_argument(
        "--mix_span_prob_start",
        type=float,
        default=0.8,
        help="starting span probability at beginning of schedule (used when schedule != fixed).",
    )
    mg.add_argument(
        "--mix_span_prob_end",
        type=float,
        default=0.2,
        help="ending span probability at end of schedule (used when schedule != fixed).",
    )
    mg.add_argument(
        "--mix_schedule_epochs",
        type=int,
        default=0,
        help="number of epochs to apply schedule over. 0 means all epochs.",
    )
    mg.add_argument(
        "--mix_min_span_prob",
        type=float,
        default=0.0,
        help="clamp: minimum span probability.",
    )
    mg.add_argument(
        "--mix_max_span_prob",
        type=float,
        default=1.0,
        help="clamp: maximum span probability.",
    )

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

    p.add_argument(
        "--no_dedupe_pairs",
        action="store_true",
        help="do not deduplicate pairs across multiple jsonl inputs (within each dataset and/or merged mode)",
    )

    p.add_argument(
        "--steps_per_epoch",
        type=int,
        default=0,
        help="override training steps per epoch. 0 = auto (sum of loaders for mixed mode; len(loader) for merged mode).",
    )

    args = p.parse_args()

    if args.pairs is None:
        if not args.pairs_sentence or not args.pairs_span:
            raise SystemExit(
                "Provide either --pairs (simple merge) OR both --pairs_sentence and --pairs_span (probabilistic mixing)."
            )
    else:
        if args.pairs_sentence or args.pairs_span:
            raise SystemExit("Use either --pairs OR (--pairs_sentence and --pairs_span), not both.")

    # Validate mixing params
    def _in01(x: float) -> bool:
        return 0.0 <= x <= 1.0

    if args.pairs is None:
        if args.mix_schedule == "fixed":
            if not _in01(args.mix_span_prob):
                raise SystemExit("--mix_span_prob must be in [0, 1].")
        else:
            if not _in01(args.mix_span_prob_start) or not _in01(args.mix_span_prob_end):
                raise SystemExit("--mix_span_prob_start/end must be in [0, 1].")
        if not _in01(args.mix_min_span_prob) or not _in01(args.mix_max_span_prob):
            raise SystemExit("--mix_min_span_prob/max must be in [0, 1].")
        if args.mix_min_span_prob > args.mix_max_span_prob:
            raise SystemExit("--mix_min_span_prob must be <= --mix_max_span_prob.")

    return args


def lr_schedule(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def _load_samples(paths: List[Path], dedupe: bool) -> List:
    for p in paths:
        if not p.exists():
            raise RuntimeError(f"pairs file not found: {p}")

    if not dedupe:
        merged: List = []
        for p in paths:
            merged.extend(read_pairs_jsonl(p))
        return merged

    seen: Set[Tuple[str, str]] = set()
    out: List = []
    for p in paths:
        ss = read_pairs_jsonl(p)
        for s in ss:
            key = (s.src, s.tgt)
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
    return out


def _cycle_iter(dl: DataLoader):
    while True:
        for batch in dl:
            yield batch


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def span_prob_for_step(
    *,
    step: int,  # 0-indexed global step across the entire training
    total_steps: int,  # total steps across the entire training
    schedule: str,
    fixed_prob: float,
    start_prob: float,
    end_prob: float,
    schedule_steps: int,  # number of steps to apply schedule over (0 => all total_steps)
    clamp_min: float,
    clamp_max: float,
) -> float:
    """
    Step-based schedule for P(span).
    - fixed: constant fixed_prob
    - linear: linearly interpolate start_prob -> end_prob over schedule_steps (or all steps if 0)
    - cosine: cosine anneal start_prob -> end_prob over schedule_steps (or all steps if 0)
    """
    if schedule == "fixed":
        return _clamp(fixed_prob, clamp_min, clamp_max)

    S = schedule_steps if schedule_steps > 0 else total_steps
    S = max(1, S)

    # progress t in [0, 1]
    if S <= 1:
        t = 1.0
    else:
        t = step / (S - 1) if step < (S - 1) else 1.0

    if schedule == "linear":
        prob = start_prob + (end_prob - start_prob) * t
    elif schedule == "cosine":
        # cosine annealing from start -> end
        # t=0 => start, t=1 => end
        prob = end_prob + 0.5 * (start_prob - end_prob) * (1.0 + math.cos(math.pi * t))
    else:
        # should not happen due to argparse choices
        prob = fixed_prob

    return _clamp(prob, clamp_min, clamp_max)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dedupe = not args.no_dedupe_pairs

    # -------------------------
    # Load + split samples
    # -------------------------
    if args.pairs is not None:
        paths = [Path(x) for x in args.pairs]
        samples = _load_samples(paths, dedupe=dedupe)
        if len(samples) < 10:
            raise RuntimeError(f"Too few samples after merge: {len(samples)}")

        train_samples, valid_samples = split_train_valid(
            samples, valid_ratio=args.valid_ratio, seed=args.seed
        )

        src_vocab = build_vocab((s.src for s in train_samples), min_freq=args.min_char_freq)
        tgt_vocab = build_vocab((s.tgt for s in train_samples), min_freq=args.min_char_freq)

        save_vocab(src_vocab, out_dir / "src_vocab.json")
        save_vocab(tgt_vocab, out_dir / "tgt_vocab.json")

        train_ds = PairDataset(
            train_samples, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len
        )
        valid_ds = PairDataset(
            valid_samples, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len
        )

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

        train_steps = args.steps_per_epoch or len(train_dl)

        # Not used in merged mode
        train_dl_sentence = None
        train_dl_span = None
        valid_dl_sentence = None
        valid_dl_span = None

    else:
        sent_paths = [Path(x) for x in args.pairs_sentence]
        span_paths = [Path(x) for x in args.pairs_span]

        sent_samples = _load_samples(sent_paths, dedupe=dedupe)
        span_samples = _load_samples(span_paths, dedupe=dedupe)

        if len(sent_samples) < 10:
            raise RuntimeError(f"Too few sentence samples: {len(sent_samples)}")
        if len(span_samples) < 10:
            raise RuntimeError(f"Too few span samples: {len(span_samples)}")

        sent_train, sent_valid = split_train_valid(
            sent_samples, valid_ratio=args.valid_ratio, seed=args.seed
        )
        span_train, span_valid = split_train_valid(
            span_samples, valid_ratio=args.valid_ratio, seed=args.seed
        )

        src_iter = itertools.chain((s.src for s in sent_train), (s.src for s in span_train))
        tgt_iter = itertools.chain((s.tgt for s in sent_train), (s.tgt for s in span_train))

        src_vocab = build_vocab(src_iter, min_freq=args.min_char_freq)
        tgt_vocab = build_vocab(tgt_iter, min_freq=args.min_char_freq)

        save_vocab(src_vocab, out_dir / "src_vocab.json")
        save_vocab(tgt_vocab, out_dir / "tgt_vocab.json")

        train_ds_sentence = PairDataset(
            sent_train, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len
        )
        train_ds_span = PairDataset(
            span_train, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len
        )
        valid_ds_sentence = PairDataset(
            sent_valid, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len
        )
        valid_ds_span = PairDataset(
            span_valid, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len
        )

        train_dl_sentence = DataLoader(
            train_ds_sentence,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
            pin_memory=(device.type == "cuda"),
        )
        train_dl_span = DataLoader(
            train_ds_span,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
            pin_memory=(device.type == "cuda"),
        )
        valid_dl_sentence = DataLoader(
            valid_ds_sentence,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
            pin_memory=(device.type == "cuda"),
        )
        valid_dl_span = DataLoader(
            valid_ds_span,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
            pin_memory=(device.type == "cuda"),
        )

        auto_steps = len(train_dl_sentence) + len(train_dl_span)
        train_steps = args.steps_per_epoch or auto_steps

        print(
            f"[mix] sentence batches={len(train_dl_sentence)} span batches={len(train_dl_span)} "
            f"steps_per_epoch={train_steps} schedule={args.mix_schedule}"
        )

        train_dl = None
        valid_dl = None

    # -------------------------
    # Model / optim
    # -------------------------
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

    # For step-based schedule we need total steps across training in mixed mode.
    # (Merged mode does not use mix schedule.)
    total_steps_all = args.epochs * train_steps

    # schedule length in steps (0 => all steps)
    if args.mix_schedule_epochs > 0:
        schedule_steps = args.mix_schedule_epochs * train_steps
    else:
        schedule_steps = 0

    if args.pairs is None:
        assert train_dl_sentence is not None and train_dl_span is not None
        it_sentence = _cycle_iter(train_dl_sentence)
        it_span = _cycle_iter(train_dl_span)

    # -------------------------
    # Train loop
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()

        if args.pairs is not None:
            assert train_dl is not None
            pbar = tqdm(train_dl, desc=f"train epoch {epoch}", dynamic_ncols=True)
            step_in_epoch = 0
            for batch in pbar:
                if step_in_epoch >= train_steps:
                    break
                step_in_epoch += 1

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

        else:
            # step-level scheduled mixing probability
            pbar = tqdm(
                range(train_steps),
                desc=f"train(mix:{args.mix_schedule}) epoch {epoch}",
                dynamic_ncols=True,
            )
            for _ in pbar:
                p_span = span_prob_for_step(
                    step=global_step,
                    total_steps=total_steps_all,
                    schedule=args.mix_schedule,
                    fixed_prob=args.mix_span_prob,
                    start_prob=args.mix_span_prob_start,
                    end_prob=args.mix_span_prob_end,
                    schedule_steps=schedule_steps,
                    clamp_min=args.mix_min_span_prob,
                    clamp_max=args.mix_max_span_prob,
                )

                use_span = (random.random() < p_span)
                batch = next(it_span) if use_span else next(it_sentence)

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
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                lr = lr_schedule(global_step, args.lr, args.warmup_steps)
                for pg in opt.param_groups:
                    pg["lr"] = lr
                opt.step()

                global_step += 1
                pbar.set_postfix(loss=float(loss.item()), lr=lr, p_span=f"{p_span:.3f}")

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            if args.pairs is not None:
                assert valid_dl is not None
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
            else:
                assert valid_dl_sentence is not None and valid_dl_span is not None
                for name, vdl in [("sent", valid_dl_sentence), ("span", valid_dl_span)]:
                    for batch in tqdm(vdl, desc=f"valid({name}) epoch {epoch}", dynamic_ncols=True):
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
