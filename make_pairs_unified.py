from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ============================================================
# Common cleanup / filters
# ============================================================
SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")
_WHITESPACE_RE = re.compile(r"[ \t]+")

# Strict: Hiragana/Katakana/Kanji + prolonged sound mark, and exclude Kan numerals.
_KAN_NUMERALS = "〇零一二三四五六七八九十百千万億兆京"
_ALLOWED_SURFACE_RE = re.compile(
    rf"^(?!.*[{_KAN_NUMERALS}])"
    r"[\u3040-\u309F"            # Hiragana
    r"\u30A0-\u30FF"             # Katakana
    r"\u4E00-\u9FFF"             # Kanji
    r"ー"                        # prolonged sound mark
    r"]+$"
)

# Reading should be hiragana + prolonged sound mark only (no spaces).
_ALLOWED_READING_RE = re.compile(r"^[\u3041-\u3096ー]+$")


def split_sentences(s: str) -> List[str]:
    if not s:
        return []
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        out.extend([x for x in SENT_SPLIT_RE.split(ln) if x])
    return out


def contains_disallowed_chars(s: str) -> bool:
    return _ALLOWED_SURFACE_RE.match(s) is None


# ============================================================
# Analyzers
# ============================================================
@dataclass(frozen=True)
class Token:
    surface: str
    reading_hira: Optional[str]  # None if unknown/unavailable


class Analyzer:
    name: str

    def tokenize(self, text: str) -> List[Token]:
        raise NotImplementedError


class NoneAnalyzer(Analyzer):
    name = "none"

    def tokenize(self, text: str) -> List[Token]:
        return [Token(surface=text, reading_hira=None)]


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
        out = []
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
            r = m.reading_form()
            if not r or r == "*":
                out.append(Token(surface=surf, reading_hira=None))
            else:
                out.append(Token(surface=surf, reading_hira=self._kata_to_hira(r)))
        return out


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
        out = []
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


# ============================================================
# Source loaders (make_pairs_from_*.py を統合)
# ============================================================
WIKI40B_MARKERS_RE = re.compile(r"(_START_ARTICLE_|_START_SECTION_|_START_PARAGRAPH_|_NEWLINE_)", flags=0)


def clean_wiki40b_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("_NEWLINE_", "\n")
    s = WIKI40B_MARKERS_RE.sub(" ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", "\n", s)
    return s.strip()


def clean_text_generic(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WHITESPACE_RE.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_aozora_text(s: str, drop_boilerplate: bool) -> str:
    s = clean_text_generic(s)
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


def iter_source_sentences(args: argparse.Namespace) -> Iterable[str]:
    """
    Yields cleaned sentences from the chosen source.
    Sources correspond to:
      - range3/wiki40b-ja
      - globis-university/aozorabunko-clean
      - y2lan/japan-law
      - Miwa-Keita/zenz-v2.5-dataset
    """
    src = args.source

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("datasets not installed. Install: pip install datasets") from e

    if src == "wiki40b_ja":
        ds = load_dataset("range3/wiki40b-ja", split=args.split, streaming=args.streaming)
        for ex in ds:
            txt = clean_wiki40b_text(str(ex.get("text", "") or ""))
            for sent in split_sentences(txt):
                yield sent
        return

    if src == "aozorabunko":
        ds = load_dataset("globis-university/aozorabunko-clean", split=args.split, streaming=args.streaming)
        for ex in ds:
            meta = ex.get("meta", {}) or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            if not isinstance(meta, dict):
                meta = {}

            if args.aozora_only_modern:
                year = meta.get("first_pub_year")
                # Some records have missing year or non-int; keep defensive.
                try:
                    y = int(year) if year is not None else None
                except Exception:
                    y = None
                # "modern" heuristic (same spirit as the original script): keep >= 1945
                if y is not None and y < 1945:
                    continue

            if args.aozora_require_pd:
                # meta["rights"] sometimes contains "public domain" markers
                rights = str(meta.get("rights", "") or "").lower()
                if not ("public" in rights and "domain" in rights):
                    continue

            text = clean_aozora_text(str(ex.get("text", "") or ""), drop_boilerplate=args.aozora_drop_boilerplate)
            for sent in split_sentences(text):
                yield sent
        return

    if src == "japan_law":
        ds = load_dataset("y2lan/japan-law", split="train", streaming=args.streaming)
        for ex in ds:
            body = str(ex.get("body", "") or "")
            title = str(ex.get("title", "") or "")
            num = str(ex.get("num", "") or "")

            parts: List[str] = []
            if args.law_include_num and num:
                parts.append(num)
            if args.law_include_title and title:
                parts.append(title)
            parts.append(body)

            txt = clean_text_generic("\n".join([p for p in parts if p]))
            for sent in split_sentences(txt):
                yield sent
        return

    if src == "zenz_v2_5":
        ds = load_dataset("Miwa-Keita/zenz-v2.5-dataset", split=args.split, streaming=args.streaming)
        for ex in ds:
            out = clean_text_generic(str(ex.get("output", "") or ""))
            if not out:
                continue
            for sent in split_sentences(out):
                yield sent
        return

    raise ValueError(f"Unknown source: {src}")


# ============================================================
# Feature extraction
# ============================================================
def tokens_to_reading(tokens: List[Token]) -> Tuple[str, float, bool]:
    """
    Returns:
      - reading_hira: concatenated reading (hiragana)
      - coverage: known_reading_tokens / total_tokens (excluding whitespace-only)
      - has_unknown: any token reading is unknown
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


def tokens_to_boundaries(tokens: List[Token]) -> Tuple[str, List[int], float, bool]:
    """
    Build boundary positions (character offsets) from token readings.

    boundaries: list of offsets in [1, len(reading_hira)-1] where a token boundary occurs.
                Example: readings=["あるつ", "はいまー", "びょう"] => reading="あるつはいまーびょう", boundaries=[2, 6]
    """
    if not tokens:
        return "", [], 0.0, True

    # Filter out whitespace tokens while preserving order.
    toks = [t for t in tokens if t.surface and (not t.surface.isspace())]
    if not toks:
        return "", [], 0.0, True

    reading, coverage, has_unknown = tokens_to_reading(toks)
    if not reading:
        return "", [], coverage, has_unknown

    boundaries: List[int] = []
    offset = 0
    for t in toks:
        if t.reading_hira is None:
            # Unknown makes boundary supervision unreliable.
            has_unknown = True
            continue
        offset += len(t.reading_hira)
        if 0 < offset < len(reading):
            boundaries.append(offset)

    # If we had any unknowns, boundaries may be incomplete; caller can drop.
    return reading, boundaries, coverage, has_unknown


def _contains_kanji(s: str) -> bool:
    for ch in s:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
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


def make_span_examples(
    toks: List[Token],
    samples_per_sentence: int,
    max_target_tokens: int,
    max_left_chars: int,
    max_right_chars: int,
    prefer_kanji_target: bool,
    rng: random.Random,
) -> List[Tuple[str, str, str, str]]:
    """
    Span sampling for KKC (same semantics as existing make_pairs_from_*).
    Returns: (left_ctx, reading_hira, right_ctx, target_surface)
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
        left = "".join(surfaces[:i])
        right = "".join(surfaces[j:])
        left_ctx = _slice_left_context(left, max_left_chars)
        right_ctx = _slice_right_context(right, max_right_chars)
        rh = "".join([r for r in readings[i:j] if r is not None])
        tgt_sf = "".join(surfaces[i:j])
        out.append((left_ctx, rh, right_ctx, tgt_sf))
    return out


# ============================================================
# CLI / Main
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified dataset builder (KKC pairs + hiragana segmentation labels).",
    )
    p.add_argument("--out", type=str, required=True, help="output jsonl")
    p.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["kkc_sentence", "kkc_span", "seg_boundary"],
        help="output format / task",
    )

    # Source
    p.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["wiki40b_ja", "aozorabunko", "japan_law", "zenz_v2_5"],
        help="dataset source",
    )
    p.add_argument("--split", type=str, default="train", help="datasets split (if applicable)")
    p.add_argument("--streaming", action="store_true", help="use datasets streaming mode")

    # Generic filtering
    p.add_argument("--max_lines", type=int, default=200000, help="stop after N outputs")
    p.add_argument("--min_len", type=int, default=4, help="min sentence length")
    p.add_argument("--max_len", type=int, default=256, help="max sentence length")
    p.add_argument("--allow_ascii", action="store_true", help="keep sentences with lots of ascii (default: filtered)")
    p.add_argument("--dedup_surface", action="store_true", help="drop duplicates by output surface")

    # Analyzer
    p.add_argument("--analyzer", type=str, default="sudachi", choices=["sudachi", "mecab", "none"])
    p.add_argument("--sudachi_mode", type=str, default="C", choices=["A", "B", "C"])
    p.add_argument("--min_coverage", type=float, default=1.0)
    p.add_argument("--drop_unknown", action="store_true", help="drop samples containing unknown readings (recommended)")

    # KKC span sampling args (used only for task=kkc_span)
    p.add_argument("--samples_per_sentence", type=int, default=2)
    p.add_argument("--max_target_tokens", type=int, default=4)
    p.add_argument("--max_left_chars", type=int, default=24)
    p.add_argument("--max_right_chars", type=int, default=24)
    p.add_argument(
        "--allow_no_kanji_target",
        action="store_true",
        help="kkc_span: allow targets without kanji (default: prefer targets containing kanji)",
    )
    p.add_argument("--rng_seed", type=int, default=42)

    # Source-specific switches (safe defaults)
    p.add_argument("--aozora_only_modern", action="store_true", help="aozorabunko: keep modern works (>=1945)")
    p.add_argument("--aozora_require_pd", action="store_true", help="aozorabunko: require public domain marker")
    p.add_argument("--aozora_drop_boilerplate", action="store_true", help="aozorabunko: drop boilerplate lines")

    p.add_argument("--law_include_title", action="store_true", help="japan_law: include law title")
    p.add_argument("--law_include_num", action="store_true", help="japan_law: include law number/id")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    rng = random.Random(args.rng_seed)
    analyzer = build_analyzer(args.analyzer, args.sudachi_mode)

    n_written = 0
    n_seen = 0
    n_filtered = 0
    n_deduped = 0
    seen_surface: set[str] = set()

    with open(args.out, "w", encoding="utf-8") as fo:
        for sent in iter_source_sentences(args):
            n_seen += 1
            s = sent.strip()
            if not s:
                continue
            if len(s) < args.min_len or len(s) > args.max_len:
                n_filtered += 1
                continue
            if contains_disallowed_chars(s):
                n_filtered += 1
                continue

            if not args.allow_ascii:
                ascii_ratio = sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))
                if ascii_ratio > 0.4:
                    n_filtered += 1
                    continue

            toks = analyzer.tokenize(s)

            if args.task == "kkc_sentence":
                reading_hira, coverage, has_unknown = tokens_to_reading(toks)
                if (not reading_hira) or (_ALLOWED_READING_RE.match(reading_hira) is None):
                    n_filtered += 1
                    continue
                if coverage < args.min_coverage:
                    n_filtered += 1
                    continue
                if args.drop_unknown and has_unknown:
                    n_filtered += 1
                    continue

                out_surface = s
                if args.dedup_surface:
                    key = out_surface.strip()
                    if key in seen_surface:
                        n_deduped += 1
                        continue
                    seen_surface.add(key)

                obj = {
                    "id": f"{args.split}:{n_written}",
                    "reading_hira": reading_hira,
                    "surface": out_surface,
                    "analyzer": analyzer.name,
                    "coverage": round(coverage, 4),
                    "mode": "sentence",
                }
                fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_written += 1

            elif args.task == "kkc_span":
                reading_hira, coverage, has_unknown = tokens_to_reading(toks)
                if (not reading_hira) or (_ALLOWED_READING_RE.match(reading_hira) is None):
                    n_filtered += 1
                    continue
                if coverage < args.min_coverage:
                    n_filtered += 1
                    continue
                if args.drop_unknown and has_unknown:
                    n_filtered += 1
                    continue

                examples = make_span_examples(
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
                    if not rh or (_ALLOWED_READING_RE.match(rh) is None):
                        n_filtered += 1
                        continue
                    if args.dedup_surface:
                        key = tgt_surface.strip()
                        if key in seen_surface:
                            n_deduped += 1
                            continue
                        seen_surface.add(key)

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

            else:  # seg_boundary
                reading_hira, boundaries, coverage, has_unknown = tokens_to_boundaries(toks)
                if (not reading_hira) or (_ALLOWED_READING_RE.match(reading_hira) is None):
                    n_filtered += 1
                    continue
                if coverage < args.min_coverage:
                    n_filtered += 1
                    continue
                # Segmentation supervision becomes ambiguous with unknown readings.
                if has_unknown:
                    n_filtered += 1
                    continue
                if len(reading_hira) < 2:
                    n_filtered += 1
                    continue

                # boundary string: length len(reading_hira)-1, each position is "1" if boundary after i-th char.
                boundary_bits = ["0"] * (len(reading_hira) - 1)
                for b in boundaries:
                    if 1 <= b <= len(reading_hira) - 1:
                        boundary_bits[b - 1] = "1"
                boundary_str = "".join(boundary_bits)

                if args.dedup_surface:
                    # For seg task, dedup by reading string (more natural).
                    key = reading_hira
                    if key in seen_surface:
                        n_deduped += 1
                        continue
                    seen_surface.add(key)

                obj = {
                    "id": f"{args.split}:{n_written}",
                    "reading_hira": reading_hira,
                    "boundaries": boundaries,
                    "boundary": boundary_str,
                    "analyzer": analyzer.name,
                    "coverage": round(coverage, 4),
                    "mode": "seg_boundary",
                }
                fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_written += 1

            if n_written >= args.max_lines:
                break

            if n_written % 5000 == 0 and n_written > 0:
                print(
                    f"written={n_written} seen={n_seen} filtered={n_filtered} deduped={n_deduped}",
                    file=sys.stderr,
                )

    print(
        f"done: out={args.out} written={n_written} seen={n_seen} filtered={n_filtered} deduped={n_deduped}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
