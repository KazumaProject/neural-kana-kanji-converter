from __future__ import annotations

"""Create rerank_train.jsonl using Sudachi/MeCab tokenization + same-reading substitutions.

Input: one or more jsonl files containing gold pairs.
  - sentence pairs: {"reading_hira": ..., "surface": ...}
  - context/span pairs: {"left": ..., "reading_hira": ..., "right": ..., "surface": ...}

Output: jsonl where each line is:
  {
    "left": "...",            # optional
    "reading_hira": "...",    # required
    "right": "...",           # optional
    "gold": "...",            # required (surface)
    "candidates": ["...", ...],  # strings only; length = topk (default 16)
    "meta": {...}             # optional info
  }

Design notes:
  - Candidates are produced by replacing some content tokens with other surfaces that share the
    same reading (built from a corpus-wide reading->surface frequency table).
  - The candidate order is by a simple log-frequency score (with replacement penalty), so the
    list is already “priority order” before reranking.
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple

# -------------------------
# Character filters
# -------------------------
_KAN_NUMERALS = "〇零一二三四五六七八九十百千万億兆京"
_ALLOWED_SURFACE_RE = re.compile(
    rf"^(?!.*[{_KAN_NUMERALS}])"
    r"[\u3040-\u309F"            # Hiragana
    r"\u30A0-\u30FF"             # Katakana
    r"\u4E00-\u9FFF"             # Kanji
    r"ー"                        # prolonged sound mark
    r"]+$"
)
_ALLOWED_READING_RE = re.compile(r"^[\u3041-\u3096ー]+$")

def contains_kanji(s: str) -> bool:
    return any(0x4E00 <= ord(ch) <= 0x9FFF for ch in s)

def contains_katakana(s: str) -> bool:
    return any(0x30A0 <= ord(ch) <= 0x30FF for ch in s)

def contains_hiragana_only(s: str) -> bool:
    return all(0x3040 <= ord(ch) <= 0x309F or ch == "ー" for ch in s)

def kata_to_hira(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 0x30A1 <= o <= 0x30F6:
            out.append(chr(o - 0x60))
        else:
            out.append(ch)
    return "".join(out)

# -------------------------
# Analyzers (copied in spirit from make_pairs_unified.py)
# -------------------------
@dataclass(frozen=True)
class Token:
    surface: str
    reading_hira: Optional[str]  # None if unknown/unavailable

class Analyzer:
    name: str
    def tokenize(self, text: str) -> List[Token]:
        raise NotImplementedError

class SudachiAnalyzer(Analyzer):
    name = "sudachi"
    def __init__(self, mode: str) -> None:
        try:
            from sudachipy import dictionary as sudachi_dictionary  # type: ignore
            from sudachipy import tokenizer as sudachi_tokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError("SudachiPy not installed. Install: pip install sudachipy sudachidict_core") from e
        self._tokenizer = sudachi_dictionary.Dictionary().create()
        self._mode = {
            "A": sudachi_tokenizer.Tokenizer.SplitMode.A,
            "B": sudachi_tokenizer.Tokenizer.SplitMode.B,
            "C": sudachi_tokenizer.Tokenizer.SplitMode.C,
        }[mode]

    def tokenize(self, text: str) -> List[Token]:
        ms = self._tokenizer.tokenize(text, self._mode)
        out: List[Token] = []
        for m in ms:
            surf = m.surface()
            r = m.reading_form()
            if not r or r == "*":
                out.append(Token(surface=surf, reading_hira=None))
            else:
                out.append(Token(surface=surf, reading_hira=kata_to_hira(r)))
        return out

class MeCabAnalyzer(Analyzer):
    name = "mecab"
    def __init__(self) -> None:
        try:
            import fugashi  # type: ignore
        except Exception as e:
            raise RuntimeError("fugashi not installed. Install: pip install fugashi[unidic-lite]") from e
        self._tagger = fugashi.Tagger()

    def tokenize(self, text: str) -> List[Token]:
        out: List[Token] = []
        for w in self._tagger(text):
            surf = w.surface
            reading = None
            try:
                kana = getattr(w.feature, "kana", None)
                if isinstance(kana, str) and kana and kana != "*":
                    reading = kata_to_hira(kana)
            except Exception:
                reading = None
            out.append(Token(surface=surf, reading_hira=reading))
        return out

def build_analyzer(name: str, sudachi_mode: str) -> Analyzer:
    name = name.lower()
    if name == "sudachi":
        return SudachiAnalyzer(mode=sudachi_mode)
    if name == "mecab":
        return MeCabAnalyzer()
    raise ValueError(f"Unknown analyzer: {name}")

# -------------------------
# IO helpers
# -------------------------
def iter_jsonl(paths: List[str], max_lines: int) -> Iterator[Dict]:
    n = 0
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                yield obj
                n += 1
                if max_lines > 0 and n >= max_lines:
                    return

def get_fields(obj: Dict) -> Tuple[str, str, str, str, Optional[str]]:
    """Return (left, reading_hira, right, surface, id)."""
    left = obj.get("left", "") or ""
    right = obj.get("right", "") or ""
    reading_hira = obj.get("reading_hira", "") or ""
    surface = obj.get("surface", "") or ""
    sid = obj.get("id", None)
    if not isinstance(sid, str):
        sid = None
    return left, reading_hira, right, surface, sid

# -------------------------
# Frequency table
# -------------------------
def is_surface_allowed(s: str) -> bool:
    # Must not contain whitespace, and use allowed Japanese blocks; exclude kan numerals.
    if not s or any(ch.isspace() for ch in s):
        return False
    return _ALLOWED_SURFACE_RE.match(s) is not None

def is_reading_allowed(r: str) -> bool:
    return bool(r) and _ALLOWED_READING_RE.match(r) is not None

def build_reading_surface_freq(
    pairs_paths: List[str],
    analyzer: Analyzer,
    max_lines: int,
    min_token_len: int,
    max_token_len: int,
) -> DefaultDict[str, Counter]:
    freq: DefaultDict[str, Counter] = defaultdict(Counter)
    seen = 0
    for obj in iter_jsonl(pairs_paths, max_lines=max_lines):
        _, _, _, surface, _ = get_fields(obj)
        if not surface:
            continue
        for t in analyzer.tokenize(surface):
            if not t.reading_hira:
                continue
            r = t.reading_hira
            if not is_reading_allowed(r):
                continue
            s = t.surface
            if not is_surface_allowed(s):
                continue
            if len(s) < min_token_len or len(s) > max_token_len:
                continue
            freq[r][s] += 1
        seen += 1
        if seen % 10000 == 0:
            print(f"[freq] processed {seen} lines", file=sys.stderr)
    return freq

# -------------------------
# Candidate generation (token substitution + beam)
# -------------------------
def token_is_replaceable(t: Token) -> bool:
    """Heuristic: replace content-ish tokens (kanji/katakana), not pure-hiragana tokens."""
    if not t.reading_hira:
        return False
    s = t.surface
    if contains_hiragana_only(s):
        return False
    return contains_kanji(s) or contains_katakana(s)

def top_surfaces_for_reading(
    freq: DefaultDict[str, Counter],
    reading: str,
    max_per_reading: int,
    min_count: int,
) -> List[Tuple[str, int]]:
    items = [(s, c) for s, c in freq.get(reading, {}).items() if c >= min_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:max_per_reading]

def beam_substitute(
    tokens: List[Token],
    freq: DefaultDict[str, Counter],
    topk: int,
    beam_size: int,
    max_per_reading: int,
    max_token_alts: int,
    min_reading_count: int,
    smooth: float,
    replace_penalty: float,
    orig_bias: float,
    ensure_gold: str,
) -> List[str]:
    """Return candidate sentence strings, sorted by score desc, size=topk (gold ensured)."""
    # Precompute options per token
    opts: List[List[Tuple[str, float]]] = []
    for t in tokens:
        orig = t.surface
        if t.reading_hira and token_is_replaceable(t):
            cand_items = top_surfaces_for_reading(
                freq, t.reading_hira, max_per_reading=max_per_reading, min_count=min_reading_count
            )
            # Build alt list: keep orig + other surfaces
            alts: List[str] = [orig]
            for s, _c in cand_items:
                if s == orig:
                    continue
                if not is_surface_allowed(s):
                    continue
                alts.append(s)
                if len(alts) >= max_token_alts:
                    break
            scored: List[Tuple[str, float]] = []
            # Score each option
            for a in alts:
                c = freq.get(t.reading_hira, {}).get(a, 0)
                score = math.log(c + smooth)
                if a != orig:
                    score -= replace_penalty
                else:
                    score += orig_bias
                scored.append((a, score))
            opts.append(scored)
        else:
            # Not replaceable: only original, no score contribution
            opts.append([(orig, 0.0)])

    # Beam search over token positions
    beams: List[Tuple[float, str]] = [(0.0, "")]
    for i, choices in enumerate(opts):
        new_beams: List[Tuple[float, str]] = []
        for base_score, prefix in beams:
            for surf, add in choices:
                new_beams.append((base_score + add, prefix + surf))
        # Keep top beam_size
        new_beams.sort(key=lambda x: -x[0])
        beams = new_beams[:beam_size]

    # Dedup and sort
    best_score: Dict[str, float] = {}
    for sc, s in beams:
        prev = best_score.get(s)
        if prev is None or sc > prev:
            best_score[s] = sc
    ranked = sorted(best_score.items(), key=lambda x: (-x[1], x[0]))
    cands = [s for s, _ in ranked]

    # Ensure gold is within output
    if ensure_gold and ensure_gold not in cands:
        # Put at end (lowest priority) to preserve “existing priority order”
        if len(cands) >= topk:
            cands = cands[: topk - 1] + [ensure_gold]
        else:
            cands = cands + [ensure_gold]

    # Truncate
    if len(cands) > topk:
        cands = cands[:topk]
    return cands

# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_pairs", type=str, nargs="+", required=True, help="input gold pairs jsonl (sentence or ctx/span)")
    p.add_argument("--out", type=str, required=True, help="output rerank_train.jsonl")
    p.add_argument("--analyzer", type=str, default="sudachi", choices=["sudachi", "mecab"])
    p.add_argument("--sudachi_mode", type=str, default="C", choices=["A", "B", "C"])

    # generation
    p.add_argument("--topk", type=int, default=16, help="number of candidates per sample (8..16 recommended)")
    p.add_argument("--beam_size", type=int, default=128, help="beam size used for candidate generation")
    p.add_argument("--max_per_reading", type=int, default=50, help="max distinct token surfaces kept per reading")
    p.add_argument("--max_token_alts", type=int, default=6, help="max alternatives per token (including original)")
    p.add_argument("--min_reading_count", type=int, default=2, help="min count for reading->surface entries used as alts")
    p.add_argument("--smooth", type=float, default=1.0, help="additive smoothing for log(count+smooth)")
    p.add_argument("--replace_penalty", type=float, default=0.35, help="penalty applied when replacing a token")
    p.add_argument("--orig_bias", type=float, default=0.10, help="small bonus for keeping original token surface")

    # token filters for the freq table
    p.add_argument("--min_token_len", type=int, default=1)
    p.add_argument("--max_token_len", type=int, default=32)

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_lines", type=int, default=0, help="0=all; otherwise limit for debugging")
    p.add_argument("--ensure_gold", action="store_true", help="force gold to be present in candidates (default on)")
    p.add_argument("--no_ensure_gold", action="store_true", help="do not force gold into candidates; may skip later")
    p.add_argument("--verbose_every", type=int, default=5000, help="log progress every N lines")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    analyzer = build_analyzer(args.analyzer, args.sudachi_mode)

    print("[1/2] building reading->surface frequency table ...", file=sys.stderr)
    freq = build_reading_surface_freq(
        args.in_pairs,
        analyzer=analyzer,
        max_lines=args.max_lines,
        min_token_len=args.min_token_len,
        max_token_len=args.max_token_len,
    )
    print(f"[freq] unique readings: {len(freq)}", file=sys.stderr)

    ensure_gold_default = True
    if args.no_ensure_gold:
        ensure_gold_default = False
    if args.ensure_gold:
        ensure_gold_default = True

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[2/2] generating rerank_train.jsonl ...", file=sys.stderr)
    total = 0
    gold_missing = 0

    with out_path.open("w", encoding="utf-8") as wf:
        for obj in iter_jsonl(args.in_pairs, max_lines=args.max_lines):
            left, reading_hira, right, gold, sid = get_fields(obj)
            if not gold:
                continue
            if not reading_hira:
                # Try reconstruct reading from analyzer (fallback).
                toks = analyzer.tokenize(gold)
                parts: List[str] = []
                ok = True
                for t in toks:
                    if not t.reading_hira or not is_reading_allowed(t.reading_hira):
                        ok = False
                        break
                    parts.append(t.reading_hira)
                if not ok:
                    continue
                reading_hira = "".join(parts)

            toks = analyzer.tokenize(gold)
            cands = beam_substitute(
                tokens=toks,
                freq=freq,
                topk=args.topk,
                beam_size=args.beam_size,
                max_per_reading=args.max_per_reading,
                max_token_alts=args.max_token_alts,
                min_reading_count=args.min_reading_count,
                smooth=args.smooth,
                replace_penalty=args.replace_penalty,
                orig_bias=args.orig_bias,
                ensure_gold=gold if ensure_gold_default else "",
            )
            if gold not in cands:
                gold_missing += 1

            rec: Dict = {
                "left": left,
                "reading_hira": reading_hira,
                "right": right,
                "gold": gold,
                "candidates": cands,
                "meta": {
                    "analyzer": analyzer.name,
                    "topk": args.topk,
                    "beam_size": args.beam_size,
                },
            }
            if sid is not None:
                rec["meta"]["source_id"] = sid

            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            if args.verbose_every > 0 and total % args.verbose_every == 0:
                print(
                    f"[out] wrote {total} lines | gold_missing_in_topk={gold_missing}",
                    file=sys.stderr,
                )

    print(f"[done] wrote {total} lines to {out_path}", file=sys.stderr)
    if total > 0:
        print(
            f"[stats] gold_missing_in_topk: {gold_missing} ({gold_missing/total:.3%})",
            file=sys.stderr,
        )

if __name__ == "__main__":
    main()
