from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import random

# -------------------------
# Wiki40B cleanup
# -------------------------
WIKI40B_MARKERS_RE = re.compile(r"(_START_ARTICLE_|_START_SECTION_|_START_PARAGRAPH_|_NEWLINE_)", flags=0)
SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")  # rough Japanese sentence split


def clean_wiki40b_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("_NEWLINE_", "\n")
    s = WIKI40B_MARKERS_RE.sub(" ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", "\n", s)
    return s.strip()


def split_sentences(s: str) -> List[str]:
    if not s:
        return []
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        out.extend([x for x in SENT_SPLIT_RE.split(ln) if x])
    return out


_KAN_NUMERALS = "〇零一二三四五六七八九十百千万億兆京"

_ALLOWED_SURFACE_RE = re.compile(
    rf"^(?!.*[{_KAN_NUMERALS}])"  # 漢数字がどこかにあったらNG
    r"[\u3040-\u309F"            # Hiragana
    r"\u30A0-\u30FF"             # Katakana
    r"\u4E00-\u9FFF"             # Kanji
    r"ー"                        # prolonged sound mark
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


# -------------------------
# Pair creation
# -------------------------
def tokens_to_reading(tokens: List[Token]) -> Tuple[str, float, bool]:
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


def _make_span_examples(
    toks: List[Token],
    samples_per_sentence: int,
    max_target_tokens: int,
    max_left_chars: int,
    max_right_chars: int,
    prefer_kanji_target: bool,
    rng: random.Random,
) -> List[Tuple[str, str, str, str]]:
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

    # 同一文内で同一 target_surface を重複させない
    local_seen_tgt: set[str] = set()

    out: List[Tuple[str, str, str, str]] = []
    for (i, j) in candidates:
        left_sf = "".join(surfaces[:i])
        right_sf = "".join(surfaces[j:])
        tgt_sf = "".join(surfaces[i:j])
        rh = "".join(readings[i:j])  # type: ignore[arg-type]

        if not rh or not tgt_sf:
            continue

        key = tgt_sf.strip()
        if key in local_seen_tgt:
            continue
        local_seen_tgt.add(key)

        left_sf = _slice_left_context(left_sf, max_left_chars)
        right_sf = _slice_right_context(right_sf, max_right_chars)

        out.append((left_sf, rh, right_sf, tgt_sf))
        if len(out) >= samples_per_sentence:
            break

    return out


def iter_wiki40b_sentences(split: str, streaming: bool) -> Iterable[str]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("datasets not installed. Install: pip install datasets") from e

    ds = load_dataset("range3/wiki40b-ja", split=split, streaming=streaming)
    for ex in ds:
        txt = clean_wiki40b_text(ex.get("text", ""))
        for sent in split_sentences(txt):
            yield sent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="output pairs.jsonl path")
    p.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--max_lines", type=int, default=200000, help="max output lines")
    p.add_argument("--max_len", type=int, default=120, help="max sentence length (surface)")
    p.add_argument("--min_len", type=int, default=2, help="min sentence length (surface)")

    p.add_argument("--mode", type=str, default="span", choices=["sentence", "span"])

    p.add_argument("--samples_per_sentence", type=int, default=2, help="span-mode: examples per sentence")
    p.add_argument("--max_target_tokens", type=int, default=4, help="span-mode: max tokens per target")
    p.add_argument("--max_left_chars", type=int, default=24, help="span-mode: keep last N chars of left context")
    p.add_argument("--max_right_chars", type=int, default=24, help="span-mode: keep first N chars of right context")
    p.add_argument(
        "--allow_no_kanji_target",
        action="store_true",
        help="span-mode: allow targets without kanji (default: prefer targets containing kanji)",
    )
    p.add_argument("--rng_seed", type=int, default=42, help="span-mode: RNG seed for span sampling")

    p.add_argument("--analyzer", type=str, default="sudachi", choices=["sudachi", "mecab", "none"])
    p.add_argument("--sudachi_mode", type=str, default="C", choices=["A", "B", "C"])

    p.add_argument("--min_coverage", type=float, default=1.0, help="min token reading coverage to keep a sample")
    p.add_argument("--drop_unknown", action="store_true", help="drop sentences containing unknown readings (recommended)")
    p.add_argument("--allow_ascii", action="store_true", help="keep sentences with lots of ascii (default: filtered)")

    # dedup
    p.add_argument(
        "--dedup_surface",
        action="store_true",
        help="drop duplicates by output 'surface' (sentence: full sentence, span: target surface)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    rng = random.Random(args.rng_seed)
    analyzer = build_analyzer(args.analyzer, args.sudachi_mode)

    out_path = args.out
    n_written = 0
    n_seen = 0
    n_filtered = 0
    n_deduped = 0

    # 出力surface（sentence: 文, span: ターゲット）で重複排除
    seen_surface: set[str] = set()

    with open(out_path, "w", encoding="utf-8") as fo:
        for sent in iter_wiki40b_sentences(split=args.split, streaming=args.streaming):
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

                if n_written >= args.max_lines:
                    break

            if n_written >= args.max_lines:
                break

            if n_written % 5000 == 0 and n_written > 0:
                print(
                    f"written={n_written} seen={n_seen} filtered={n_filtered} deduped={n_deduped}",
                    file=sys.stderr,
                )

    print(
        f"done: out={out_path} written={n_written} seen={n_seen} filtered={n_filtered} deduped={n_deduped}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
