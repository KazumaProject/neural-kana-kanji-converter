from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

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
        parts = [p.strip() for p in SENT_SPLIT_RE.split(ln) if p.strip()]
        out.extend(parts)
    return out


# -------------------------
# Strict surface filter (Japanese-only)
# -------------------------
# 目的:
# - surface に英字/ギリシャ文字/数字/括弧/ハイフン等を含む文を除外する
# - 許可する文字をホワイトリスト方式で定義し、それ以外が1文字でもあれば落とす
#
# 許可:
# - ひらがな: 3040-309F
# - カタカナ: 30A0-30FF
# - 漢字(CJK統合漢字): 4E00-9FFF
# - 々: 3005
# - ー: 30FC
# - ゝゞ: 309D-309E
# - ヽヾ: 30FD-30FE
# - ・: 30FB
# - 句読点/！？: 。、！？ + 半角 !?
_ALLOWED_EXTRA_CODEPOINTS = {
    # 0x3005,  # 々
    # 0x30FC,  # ー
    # 0x30FB,  # ・
    # 0x3002,  # 。
    # 0x3001,  # 、
    # 0xFF01,  # ！
    # 0xFF1F,  # ？
    # 0x0021,  # !
    # 0x003F,  # ?
}


def _is_allowed_japanese_char(ch: str) -> bool:
    if ch.isspace():
        return True

    code = ord(ch)

    # Hiragana
    if 0x3040 <= code <= 0x309F:
        return True
    # Katakana
    if 0x30A0 <= code <= 0x30FF:
        return True
    # Kanji (CJK Unified Ideographs)
    if 0x4E00 <= code <= 0x9FFF:
        return True
    # Iteration marks
    if 0x309D <= code <= 0x309E:  # ゝゞ
        return True
    if 0x30FD <= code <= 0x30FE:  # ヽヾ
        return True

    if code in _ALLOWED_EXTRA_CODEPOINTS:
        return True

    return False


def contains_disallowed_chars(s: str) -> bool:
    for ch in s:
        if not _is_allowed_japanese_char(ch):
            return True
    return False


# -------------------------
# Reading normalization
# -------------------------
_KATAKANA_RANGE = (0x30A1, 0x30F6)  # small a .. small ke


def katakana_to_hiragana(s: str) -> str:
    out: List[str] = []
    for ch in s:
        code = ord(ch)
        if _KATAKANA_RANGE[0] <= code <= _KATAKANA_RANGE[1]:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


_HIRAGANA_RE = re.compile(r"^[\u3041-\u3096ー]+$")  # include prolonged sound mark
_KATAKANA_RE = re.compile(r"^[\u30A1-\u30F6ー]+$")
_ASCII_RE = re.compile(r"^[\u0000-\u007F]+$")


def is_readable_surface_as_reading(surface: str) -> bool:
    # If token is already kana or ascii symbol/number, we can treat it as "reading".
    # NOTE: 文レベルで strict filter をかけるので、ここは既存仕様を維持。
    if not surface:
        return False
    if _HIRAGANA_RE.match(surface):
        return True
    if _KATAKANA_RE.match(surface):
        return True
    if _ASCII_RE.match(surface):
        return True
    return False


def surface_to_reading_hira(surface: str) -> str:
    # Use surface as reading if it's kana/ascii. Katakana -> hiragana
    if _KATAKANA_RE.match(surface):
        return katakana_to_hiragana(surface)
    return surface


# -------------------------
# Analyzer interface
# -------------------------
@dataclass(frozen=True)
class Token:
    surface: str
    reading_hira: Optional[str]  # None if unknown


class Analyzer:
    name: str

    def tokenize(self, text: str) -> List[Token]:
        raise NotImplementedError


class NoneAnalyzer(Analyzer):
    name = "none"

    def tokenize(self, text: str) -> List[Token]:
        # no morphological analysis: treat each char as token
        # reading == surface (hiragana normalization for katakana)
        toks: List[Token] = []
        for ch in text:
            if ch.isspace():
                continue
            rh = surface_to_reading_hira(ch) if is_readable_surface_as_reading(ch) else None
            toks.append(Token(surface=ch, reading_hira=rh))
        return toks


class SudachiAnalyzer(Analyzer):
    name = "sudachi"

    def __init__(self, mode: str = "C") -> None:
        try:
            from sudachipy import dictionary as sudachi_dictionary  # type: ignore
            from sudachipy import tokenizer as sudachi_tokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "SudachiPy not installed. Install: pip install sudachipy sudachidict_core"
            ) from e

        self._tokenizer = sudachi_dictionary.Dictionary().create()
        self._mode = {
            "A": sudachi_tokenizer.Tokenizer.SplitMode.A,
            "B": sudachi_tokenizer.Tokenizer.SplitMode.B,
            "C": sudachi_tokenizer.Tokenizer.SplitMode.C,
        }.get(mode.upper(), sudachi_tokenizer.Tokenizer.SplitMode.C)

    def tokenize(self, text: str) -> List[Token]:
        ms = self._tokenizer.tokenize(text, self._mode)
        out: List[Token] = []
        for m in ms:
            surf = m.surface()
            # Sudachi reading_form() is usually katakana for Japanese words; can be '*' sometimes.
            r = m.reading_form()
            if r and r != "*" and r != surf:
                rh = katakana_to_hiragana(r)
            else:
                rh = surface_to_reading_hira(surf) if is_readable_surface_as_reading(surf) else None
            out.append(Token(surface=surf, reading_hira=rh))
        return out


class MeCabAnalyzer(Analyzer):
    name = "mecab"

    def __init__(self) -> None:
        """
        Uses fugashi.
        Recommended minimal install:
          pip install fugashi unidic-lite
        """
        try:
            from fugashi import Tagger  # type: ignore
        except Exception as e:
            raise RuntimeError("fugashi not installed. Install: pip install fugashi unidic-lite") from e

        # Let fugashi pick default dic (unidic-lite if installed).
        self._tagger = Tagger()

    def _get_reading(self, w) -> Optional[str]:
        # Try a few common fugashi feature shapes:
        # - UniDic: w.feature has .kana / .pron / .reading (varies)
        # - IPADIC: w.feature is tuple/list with reading at index 7
        feat = getattr(w, "feature", None)

        # UniDic-like: attributes
        for attr in ("kana", "reading", "pron"):
            val = getattr(feat, attr, None)
            if isinstance(val, str) and val and val != "*":
                return katakana_to_hiragana(val)

        # IPADIC-like: sequence
        if isinstance(feat, (tuple, list)):
            # ipadic: ... , reading, pronunciation
            if len(feat) >= 8:
                val = feat[7]
                if isinstance(val, str) and val and val != "*":
                    return katakana_to_hiragana(val)

        return None

    def tokenize(self, text: str) -> List[Token]:
        out: List[Token] = []
        for w in self._tagger(text):
            surf = str(w.surface)
            r = self._get_reading(w)
            if r:
                rh = r
            else:
                rh = surface_to_reading_hira(surf) if is_readable_surface_as_reading(surf) else None
            out.append(Token(surface=surf, reading_hira=rh))
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
      has_unknown: whether any token reading is unknown for tokens that are not "skippable"
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
            # TODO: if you integrate a learned estimator, fill reading here.
            continue
        known += 1
        parts.append(t.reading_hira)

    reading = "".join(parts)
    coverage = known / max(1, total)
    return reading, coverage, has_unknown


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

    p.add_argument("--analyzer", type=str, default="sudachi", choices=["sudachi", "mecab", "none"])
    p.add_argument("--sudachi_mode", type=str, default="C", choices=["A", "B", "C"])

    p.add_argument("--min_coverage", type=float, default=1.0, help="min token reading coverage to keep a sample")
    p.add_argument("--drop_unknown", action="store_true", help="drop sentences containing unknown readings (recommended)")
    p.add_argument("--allow_ascii", action="store_true", help="keep sentences with lots of ascii (default: filtered)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    analyzer = build_analyzer(args.analyzer, args.sudachi_mode)

    out_path = args.out
    n_written = 0
    n_seen = 0
    n_filtered = 0

    with open(out_path, "w", encoding="utf-8") as fo:
        for sent in iter_wiki40b_sentences(split=args.split, streaming=args.streaming):
            n_seen += 1
            s = sent.strip()
            if not s:
                continue
            if len(s) < args.min_len or len(s) > args.max_len:
                n_filtered += 1
                continue

            # ★ 追加: surface を日本語ホワイトリストで厳格フィルタ
            # 例: "Β (お笑い芸人)" / "Β-セクレターゼ1" はここで落ちる
            if contains_disallowed_chars(s):
                n_filtered += 1
                continue

            if not args.allow_ascii:
                # rough filter: too much ascii tends to be noise for Japanese IME training
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

            obj = {
                "id": f"{args.split}:{n_written}",
                "reading_hira": reading_hira,
                "surface": s,
                "analyzer": analyzer.name,
                "coverage": round(coverage, 4),
            }
            fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1

            if n_written >= args.max_lines:
                break

            if n_written % 5000 == 0:
                print(f"written={n_written} seen={n_seen} filtered={n_filtered}", file=sys.stderr)

    print(f"done: out={out_path} written={n_written} seen={n_seen} filtered={n_filtered}", file=sys.stderr)


if __name__ == "__main__":
    main()
