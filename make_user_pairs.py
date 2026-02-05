from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple


# -------------------------
# Validation / normalization
# -------------------------
_HIRA_ALLOWED_RE = re.compile(r"^[\u3041-\u3096ー\s]+$")  # hiragana + prolonged + spaces
_WHITESPACE_RE = re.compile(r"\s+")


def norm_surface(s: str) -> str:
    s = s.strip()
    s = _WHITESPACE_RE.sub(" ", s)
    return s


def norm_reading_hira(s: str) -> str:
    s = s.strip()
    s = _WHITESPACE_RE.sub(" ", s)
    return s


def validate_pair(reading_hira: str, surface: str, allow_non_hira: bool) -> Optional[str]:
    if not reading_hira or not surface:
        return "reading_hira/surface is empty"
    if not allow_non_hira and not _HIRA_ALLOWED_RE.match(reading_hira):
        return "reading_hira contains non-hiragana chars (use --allow_non_hira to bypass)"
    if len(reading_hira) < 1 or len(reading_hira) > 512:
        return "reading_hira length out of range"
    if len(surface) < 1 or len(surface) > 512:
        return "surface length out of range"
    return None


def _looks_like_hiragana_reading(s: str) -> bool:
    # Heuristic for 3-column disambiguation.
    # Accept empty as "not reading".
    s2 = norm_reading_hira(s)
    if not s2:
        return False
    return _HIRA_ALLOWED_RE.match(s2) is not None


# -------------------------
# IO helpers
# -------------------------
def read_jsonl_pairs(path: Path) -> Iterator[Tuple[str, str]]:
    if not path.exists():
        return
        yield  # for typing
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # Support both legacy and context-aware user pairs.
            # - legacy: {reading_hira, surface}
            # - context-aware: {left, reading_hira, right, surface}
            rh = obj.get("reading_hira")
            sf = obj.get("surface")
            if isinstance(rh, str) and isinstance(sf, str):
                yield (rh, sf)


def iter_txt_pairs(path: Path, sep: str) -> Iterator[Tuple[str, str, str, str]]:
    """
    Flexible TXT formats (per line), depending on column count:

    - 2 cols:
        reading<sep>surface
      -> left/right = ""

    - 3 cols (either side context missing):
        left<sep>reading<sep>surface          (right = "")
        reading<sep>right<sep>surface         (left = "")
      -> Disambiguate by checking which column "looks like" hiragana reading.

    - 4+ cols:
        left<sep>reading<sep>right<sep>surface
      -> If more than 4, remainder is joined into surface (useful for comma; also works for tab/space).

    Notes:
    - For sep="tab" and sep="comma", empty columns are preserved (e.g. "\treading\t\tsurface").
    - For sep="space", empty columns cannot be represented; use 2/3/4 tokens separated by whitespace.
    """
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue

            if sep == "tab":
                parts = line.split("\t")  # preserves empty columns
            elif sep == "comma":
                parts = line.split(",")  # preserves empty columns
            else:
                # "space": tokens; cannot represent empty columns
                parts = line.strip().split()

            # Normalize raw parts (do NOT drop empties for tab/comma).
            parts = [p.strip() for p in parts]

            if len(parts) < 2:
                print(f"[skip] invalid line {i}: {line}", file=sys.stderr)
                continue

            if len(parts) == 2:
                # reading, surface
                reading_hira = parts[0]
                surface = parts[1]
                yield ("", reading_hira, "", surface)
                continue

            if len(parts) == 3:
                a, b, c = parts[0], parts[1], parts[2]

                # Try to infer which one is reading_hira.
                # Pattern A: left, reading, surface  (right missing)
                # Pattern B: reading, right, surface (left missing)
                a_is_reading = _looks_like_hiragana_reading(a)
                b_is_reading = _looks_like_hiragana_reading(b)

                if b_is_reading and not a_is_reading:
                    # left, reading, surface
                    left = a
                    reading_hira = b
                    right = ""
                    surface = c
                    yield (left, reading_hira, right, surface)
                    continue

                if a_is_reading and not b_is_reading:
                    # reading, right, surface
                    left = ""
                    reading_hira = a
                    right = b
                    surface = c
                    yield (left, reading_hira, right, surface)
                    continue

                # Ambiguous (both look like reading, or neither does).
                # Default to: left, reading, surface (right missing)
                # This matches the common "left\treading\tsurface" intent.
                left = a
                reading_hira = b
                right = ""
                surface = c
                yield (left, reading_hira, right, surface)
                continue

            # 4+ columns: left, reading, right, surface...
            left = parts[0]
            reading_hira = parts[1]
            right = parts[2]
            if sep == "comma":
                surface = ",".join(parts[3:]).strip()
            elif sep == "tab":
                surface = "\t".join(parts[3:]).strip()
            else:
                # "space": surface is single token at parts[3] typically, but join just in case
                surface = " ".join(parts[3:]).strip()
            yield (left, reading_hira, right, surface)


# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="output user_pairs.jsonl")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--add", action="store_true", help="add one pair from CLI args")
    g.add_argument("--from_txt", type=str, help="batch add from txt file")

    # context-aware fields
    p.add_argument("--left", type=str, default="", help="left context (optional)")
    p.add_argument("--right", type=str, default="", help="right context (optional)")
    p.add_argument("--reading_hira", type=str, help="hiragana reading (for --add)")
    p.add_argument("--surface", type=str, help="target surface (for --add)")

    p.add_argument("--sep", type=str, default="tab", choices=["tab", "comma", "space"], help="txt separator")

    p.add_argument("--overwrite", action="store_true", help="overwrite output file (default: append)")
    p.add_argument("--no_dedupe", action="store_true", help="do not deduplicate")
    p.add_argument("--allow_non_hira", action="store_true", help="allow non-hiragana in reading_hira")
    p.add_argument("--tag", type=str, default="user", help="tag stored in jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    if args.add:
        if args.reading_hira is None or args.surface is None:
            raise SystemExit("--add requires --reading_hira and --surface")
        incoming: List[Tuple[str, str, str, str]] = [(args.left or "", args.reading_hira, args.right or "", args.surface)]
    else:
        txt_path = Path(args.from_txt)
        if not txt_path.exists():
            raise SystemExit(f"not found: {txt_path}")
        incoming = list(iter_txt_pairs(txt_path, sep=args.sep))

    # normalize + validate
    normalized: List[Tuple[str, str, str, str]] = []
    n_invalid = 0
    for left, rh, right, sf in incoming:
        left2 = norm_surface(left)
        right2 = norm_surface(right)
        rh2 = norm_reading_hira(rh)
        sf2 = norm_surface(sf)
        err = validate_pair(rh2, sf2, allow_non_hira=args.allow_non_hira)
        if err:
            n_invalid += 1
            print(f"[skip] invalid pair: (left={left2!r}, rh={rh2!r}, right={right2!r}, sf={sf2!r}) reason={err}", file=sys.stderr)
            continue
        normalized.append((left2, rh2, right2, sf2))

    if not normalized:
        raise SystemExit("no valid pairs to write")

    # determine write mode
    if args.overwrite:
        existing_set: set[Tuple[str, str, str, str]] = set()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.no_dedupe:
            existing_set = set()
        else:
            # Dedupe by full 4-tuple (left, reading, right, surface).
            # (We keep legacy 2-tuple files compatible by treating missing ctx as empty strings.)
            existing_set = set()
            for rh, sf in read_jsonl_pairs(out_path):
                existing_set.add(("", rh, "", sf))

    n_written = 0
    n_dup = 0
    with out_path.open("a", encoding="utf-8") as fo:
        for left, rh, right, sf in normalized:
            if not args.no_dedupe:
                key = (left, rh, right, sf)
                if key in existing_set:
                    n_dup += 1
                    continue
                existing_set.add(key)

            obj = {
                "id": f"{args.tag}:{rh}:{sf}",
                "left": left,
                "reading_hira": rh,
                "right": right,
                "surface": sf,
                "tag": args.tag,
            }
            fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1

    print(
        f"done: out={out_path} written={n_written} dup_skipped={n_dup} invalid_skipped={n_invalid}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
