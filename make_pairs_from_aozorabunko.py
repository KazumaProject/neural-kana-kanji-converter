from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -------------------------
# Aozora cleanup / sentence split
# -------------------------
SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")  # rough Japanese sentence split
_WHITESPACE_RE = re.compile(r"[ \t]+")


def clean_aozora_text(s: str, drop_boilerplate: bool) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WHITESPACE_RE.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    if not drop_boilerplate:
        return s

    # Keep conservative to avoid removing real content
    drop_prefixes = (
        "底本：",
        "初出：",
        "入力：",
        "校正：",
        "青空文庫作成ファイル：",
        "このファイルは、インターネットの図書館、青空文庫",
        "※",
    )

    lines: List[str] = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            continue
        if any(t.startswith(p) for p in drop_prefixes):
            continue
        lines.append(t)
    return "\n".join(lines).strip()


def split_sentences(s: str) -> List[str]:
    if not s:
        return []
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        out.extend([x for x in SENT_SPLIT_RE.split(ln) if x])
    return out


# -------------------------
# Strict surface filter (avoid ASCII/marks)
# -------------------------
_ALLOWED_SURFACE_RE = re.compile(
    r"^[\u3000-\u303F"  # CJK symbols & punctuation
    r"\u3040-\u309F"    # Hiragana
    r"\u30A0-\u30FF"    # Katakana
    r"\u4E00-\u9FFF"    # Kanji
    r"\uFF01-\uFF60"    # Full-width forms (some punctuation)
    r"\s"
    r"ー・、。！？「」『』（）［］【】〔〕〈〉《》…"
    r"]+$"
)


def contains_disallowed_chars(s: str) -> bool:
    return _ALLOWED_SURFACE_RE.match(s) is None


# -------------------------
# Analyzer interface
# -------------------------
@dataclass(frozen=True)
class Token:
    surface: str
    reading_hira: Optional[str]  # None if unknown/unavailable


class Analyzer:
    name: str

    def tokenize(self, text: str) -> List[Token]:
        raise NotImplementedError


# -------------------------
# None analyzer: fallback (no reading)
# -------------------------
class NoneAnalyzer(Analyzer):
    name = "none"

    def tokenize(self, text: str) -> List[Token]:
        return [Token(surface=text, reading_hira=None)]


# -------------------------
# SudachiPy analyzer
# -------------------------
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

    @staticmethod
    def _kata_to_hira(s: str) -> str:
        out: List[str] = []
        for ch in s:
            o = ord(ch)
            if 0x30A1 <= o <= 0x30F6:
                out.append(chr(o - 0x60))
            else:
                out.append(ch)
        return "".join(out)

    def tokenize(self, text: str) -> List[Token]:
        ms = self._tokenizer.tokenize(text, self._mode)
        out: List[Token] = []
        for m in ms:
            surf = m.surface()
            r = m.reading_form()  # katakana; sometimes '*' or empty
            if not r or r == "*":
                out.append(Token(surface=surf, reading_hira=None))
            else:
                out.append(Token(surface=surf, reading_hira=self._kata_to_hira(r)))
        return out


# -------------------------
# MeCab analyzer (optional)
# -------------------------
class MeCabAnalyzer(Analyzer):
    name = "mecab"

    def __init__(self) -> None:
        try:
            import fugashi  # type: ignore
        except Exception as e:
            raise RuntimeError("fugashi not installed. Install: pip install fugashi[unidic-lite]") from e
        self._tagger = fugashi.Tagger()

    @staticmethod
    def _kata_to_hira(s: str) -> str:
        out: List[str] = []
        for ch in s:
            o = ord(ch)
            if 0x30A1 <= o <= 0x30F6:
                out.append(chr(o - 0x60))
            else:
                out.append(ch)
        return "".join(out)

    def tokenize(self, text: str) -> List[Token]:
        out: List[Token] = []
        for w in self._tagger(text):
            surf = w.surface
            reading = None
            try:
                # fugashi feature depends on dictionary; unidic-lite has kana
                kana = getattr(w.feature, "kana", None)
                if isinstance(kana, str) and kana and kana != "*":
                    reading = self._kata_to_hira(kana)
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
    if name == "none":
        return NoneAnalyzer()
    raise ValueError(f"Unknown analyzer: {name}")


# -------------------------
# Pair creation
# -------------------------
def tokens_to_reading(tokens: List[Token]) -> Tuple[str, float, bool]:
    """
    Returns:
      reading_hira: concatenated reading (skip tokens with unknown reading by inserting nothing)
      coverage: known_tokens / total_tokens
      has_unknown: whether any token reading is unknown
    """
    if not tokens:
        return "", 0.0, True

    total = 0
    known = 0
    has_unknown = False
    parts: List[str] = []

    for t in tokens:
        if not t.surface or t.surface.isspace():
            continue
        total += 1
        if t.reading_hira is None:
            has_unknown = True
            continue
        known += 1
        parts.append(t.reading_hira)

    reading = "".join(parts)
    coverage = known / max(1, total)
    return reading, coverage, has_unknown


def _contains_kanji(s: str) -> bool:
    for ch in s:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:
            return True
    return False


def _slice_left_context(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return s[-max_chars:]


def _slice_right_context(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return s[:max_chars]


def _make_span_examples(
    toks: List[Token],
    samples_per_sentence: int,
    max_target_tokens: int,
    max_left_chars: int,
    max_right_chars: int,
    prefer_kanji_target: bool,
    rng: random.Random,
) -> List[Tuple[str, str, str, str]]:
    """Create IME-like training samples from a tokenized sentence.

    Returns list of tuples:
      (left_context_surface, reading_hira, right_context_surface, target_surface)
    """
    tokens = [t for t in toks if t.surface and (not t.surface.isspace())]
    if not tokens:
        return []

    n = len(tokens)
    surfaces = [t.surface for t in tokens]
    readings = [t.reading_hira for t in tokens]

    candidates: List[Tuple[int, int]] = []
    for i in range(n):
        if readings[i] is None:
            continue
        for j in range(i + 1, min(n, i + max_target_tokens) + 1):
            if any(r is None for r in readings[i:j]):
                break
            tgt_sf = "".join(surfaces[i:j])
            if prefer_kanji_target and (not _contains_kanji(tgt_sf)):
                continue
            candidates.append((i, j))

    if not candidates:
        return []

    rng.shuffle(candidates)
    out: List[Tuple[str, str, str, str]] = []
    for (i, j) in candidates[:samples_per_sentence]:
        left_sf = "".join(surfaces[:i])
        right_sf = "".join(surfaces[j:])
        tgt_sf = "".join(surfaces[i:j])
        rh = "".join(readings[i:j])  # type: ignore[arg-type]

        left_sf = _slice_left_context(left_sf, max_left_chars)
        right_sf = _slice_right_context(right_sf, max_right_chars)

        if not rh or not tgt_sf:
            continue
        out.append((left_sf, rh, right_sf, tgt_sf))
    return out


# -------------------------
# Aozora dataset iteration
# -------------------------
def iter_aozorabunko_books(
    split: str,
    streaming: bool,
    only_modern: bool,
    require_pd: bool,
) -> Iterable[Dict[str, Any]]:
    """
    Yields each book record from globis-university/aozorabunko-clean.
    We use meta only for filtering (NOT output).
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("datasets not installed. Install: pip install datasets") from e

    ds = load_dataset("globis-university/aozorabunko-clean", split=split, streaming=streaming)
    for ex in ds:
        meta = ex.get("meta", {}) or {}
        if isinstance(meta, str):
            # defensive: meta could be stringified json
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        if only_modern:
            if str(meta.get("文字遣い種別", "")) != "新字新仮名":
                continue

        if require_pd:
            if str(meta.get("作品著作権フラグ", "")) not in ("なし", "無し"):
                continue

        yield ex


def iter_aozorabunko_sentences(
    split: str,
    streaming: bool,
    only_modern: bool,
    require_pd: bool,
    drop_boilerplate: bool,
) -> Iterable[str]:
    for ex in iter_aozorabunko_books(split, streaming, only_modern, require_pd):
        text = clean_aozora_text(str(ex.get("text", "")), drop_boilerplate=drop_boilerplate)
        for sent in split_sentences(text):
            yield sent


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="output pairs.jsonl path")

    # aozora source controls
    p.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    p.add_argument("--streaming", action="store_true", help="streaming load (low memory)")
    p.add_argument("--only_modern", action="store_true", help="keep only 新字新仮名 (modern Japanese)")
    p.add_argument("--require_public_domain", action="store_true", help="keep only 作品著作権フラグ == なし")
    p.add_argument("--drop_boilerplate", action="store_true", help="drop common Aozora boilerplate lines")

    # sampling/output controls
    p.add_argument("--max_lines", type=int, default=200000, help="max output lines")
    p.add_argument("--max_sents_seen", type=int, default=0, help="stop after seeing N sentences (0=unlimited)")
    p.add_argument("--max_len", type=int, default=120, help="max sentence length (surface)")
    p.add_argument("--min_len", type=int, default=2, help="min sentence length (surface)")

    # mode
    p.add_argument("--mode", type=str, default="span", choices=["sentence", "span"])

    # span-mode controls
    p.add_argument("--samples_per_sentence", type=int, default=2)
    p.add_argument("--max_target_tokens", type=int, default=4)
    p.add_argument("--max_left_chars", type=int, default=24)
    p.add_argument("--max_right_chars", type=int, default=24)
    p.add_argument("--allow_no_kanji_target", action="store_true")
    p.add_argument("--rng_seed", type=int, default=42)

    # analyzer
    p.add_argument("--analyzer", type=str, default="sudachi", choices=["sudachi", "mecab", "none"])
    p.add_argument("--sudachi_mode", type=str, default="C", choices=["A", "B", "C"])

    # filtering
    p.add_argument("--min_coverage", type=float, default=1.0)
    p.add_argument("--drop_unknown", action="store_true", help="drop sentences containing unknown readings")
    p.add_argument("--allow_ascii", action="store_true", help="keep sentences with lots of ascii (default: filtered)")
    p.add_argument("--no_strict_surface", action="store_true", help="disable strict surface whitelist")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.rng_seed)
    analyzer = build_analyzer(args.analyzer, args.sudachi_mode)

    n_written = 0
    n_seen = 0
    n_filtered = 0

    with open(args.out, "w", encoding="utf-8") as fo:
        for sent in iter_aozorabunko_sentences(
            split=args.split,
            streaming=args.streaming,
            only_modern=args.only_modern,
            require_pd=args.require_public_domain,
            drop_boilerplate=args.drop_boilerplate,
        ):
            n_seen += 1
            if args.max_sents_seen > 0 and n_seen > args.max_sents_seen:
                break

            s = sent.strip()
            if not s:
                continue
            if len(s) < args.min_len or len(s) > args.max_len:
                n_filtered += 1
                continue

            if (not args.no_strict_surface) and contains_disallowed_chars(s):
                n_filtered += 1
                continue

            if not args.allow_ascii:
                ascii_ratio = sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))
                if ascii_ratio > 0.4:
                    n_filtered += 1
                    continue

            toks = analyzer.tokenize(s)
            reading_hira, coverage, has_unknown = tokens_to_reading(toks)

            if not reading_hira:
                n_filtered += 1
                continue
            if coverage < args.min_coverage:
                n_filtered += 1
                continue
            if args.drop_unknown and has_unknown:
                n_filtered += 1
                continue

            if args.mode == "sentence":
                obj = {
                    "id": f"{args.split}:{n_written}",
                    "reading_hira": reading_hira,
                    "surface": s,
                    "analyzer": analyzer.name,
                    "coverage": round(coverage, 4),
                    "mode": "sentence",
                }
                fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_written += 1
            else:
                examples = _make_span_examples(
                    toks=toks,
                    samples_per_sentence=args.samples_per_sentence,
                    max_target_tokens=args.max_target_tokens,
                    max_left_chars=args.max_left_chars,
                    max_right_chars=args.max_right_chars,
                    prefer_kanji_target=(not args.allow_no_kanji_target),
                    rng=rng,
                )
                if not examples:
                    n_filtered += 1
                    continue

                for (left_ctx, rh, right_ctx, tgt_surface) in examples:
                    obj = {
                        "id": f"{args.split}:{n_written}",
                        "left": left_ctx,
                        "reading_hira": rh,
                        "right": right_ctx,
                        "surface": tgt_surface,
                        "analyzer": analyzer.name,
                        "coverage": round(coverage, 4),
                        "mode": "span",
                    }
                    fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    n_written += 1
                    if n_written >= args.max_lines:
                        break

                if n_written >= args.max_lines:
                    break

            if n_written >= args.max_lines:
                break

            if n_written % 5000 == 0 and n_written > 0:
                print(f"written={n_written} seen={n_seen} filtered={n_filtered}", file=sys.stderr)

    print(f"done: out={args.out} written={n_written} seen={n_seen} filtered={n_filtered}", file=sys.stderr)


if __name__ == "__main__":
    main()
