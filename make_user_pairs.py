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
    # normalize spaces (keep Japanese text as-is)
    s = s.strip()
    s = _WHITESPACE_RE.sub(" ", s)
    return s


def norm_reading_hira(s: str) -> str:
    # normalize spaces, keep hiragana as-is (optionally checked)
    s = s.strip()
    s = _WHITESPACE_RE.sub(" ", s)
    return s


def validate_pair(reading_hira: str, surface: str, allow_non_hira: bool) -> Optional[str]:
    if not reading_hira or not surface:
        return "reading_hira/surface is empty"
    if not allow_non_hira and not _HIRA_ALLOWED_RE.match(reading_hira):
        return "reading_hira contains non-hiragana chars (use --allow_non_hira to bypass)"
    # very basic sanity: avoid absurdly short/long
    if len(reading_hira) < 1 or len(reading_hira) > 512:
        return "reading_hira length out of range"
    if len(surface) < 1 or len(surface) > 512:
        return "surface length out of range"
    return None


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
            rh = obj.get("reading_hira")
            sf = obj.get("surface")
            if isinstance(rh, str) and isinstance(sf, str):
                yield (rh, sf)


def load_existing_set(path: Path) -> set[Tuple[str, str]]:
    s: set[Tuple[str, str]] = set()
    for rh, sf in read_jsonl_pairs(path):
        s.add((rh, sf))
    return s


def iter_txt_pairs(path: Path, sep: str) -> Iterator[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            if sep == "tab":
                parts = line.split("\t")
            elif sep == "comma":
                parts = line.split(",")
            else:
                # "space": first token is reading, rest is surface
                parts = line.strip().split()
                if len(parts) >= 2:
                    yield (parts[0], " ".join(parts[1:]))
                    continue
                parts = []
            if len(parts) < 2:
                print(f"[skip] invalid line {i}: {line}", file=sys.stderr)
                continue
            reading_hira = parts[0].strip()
            surface = sep.join(parts[1:]).strip() if sep in ("comma",) else parts[1].strip()
            yield (reading_hira, surface)


# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="output user_pairs.jsonl")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--add", action="store_true", help="add one pair from CLI args")
    g.add_argument("--from_txt", type=str, help="batch add from txt file")

    p.add_argument("--reading_hira", type=str, help="hiragana reading (for --add)")
    p.add_argument("--surface", type=str, help="surface string (for --add)")

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
        incoming: List[Tuple[str, str]] = [(args.reading_hira, args.surface)]
    else:
        txt_path = Path(args.from_txt)
        if not txt_path.exists():
            raise SystemExit(f"not found: {txt_path}")
        incoming = list(iter_txt_pairs(txt_path, sep=args.sep))

    # normalize + validate
    normalized: List[Tuple[str, str]] = []
    n_invalid = 0
    for rh, sf in incoming:
        rh2 = norm_reading_hira(rh)
        sf2 = norm_surface(sf)
        err = validate_pair(rh2, sf2, allow_non_hira=args.allow_non_hira)
        if err:
            n_invalid += 1
            print(f"[skip] invalid pair: ({rh2!r}, {sf2!r}) reason={err}", file=sys.stderr)
            continue
        normalized.append((rh2, sf2))

    if not normalized:
        raise SystemExit("no valid pairs to write")

    # determine write mode
    if args.overwrite:
        existing_set: set[Tuple[str, str]] = set()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # truncate
        out_path.write_text("", encoding="utf-8")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing_set = set() if args.no_dedupe else load_existing_set(out_path)

    n_written = 0
    n_dup = 0
    with out_path.open("a", encoding="utf-8") as fo:
        for rh, sf in normalized:
            if not args.no_dedupe:
                key = (rh, sf)
                if key in existing_set:
                    n_dup += 1
                    continue
                existing_set.add(key)

            obj = {
                "id": f"{args.tag}:{rh}:{sf}",  # stable-ish; you can change if you prefer
                "reading_hira": rh,
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