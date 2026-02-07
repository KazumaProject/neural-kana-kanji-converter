from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List, Tuple
import re

# ============================================================
# Filters (same spirit as your project)
# ============================================================

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

_DEFAULT_MOZC_RAW_BASE = "https://raw.githubusercontent.com/google/mozc/master/src/data/dictionary_oss"


def kata_to_hira(s: str) -> str:
    out: List[str] = []
    for ch in s:
        o = ord(ch)
        if 0x30A1 <= o <= 0x30F6:
            out.append(chr(o - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def mozc_filename(i: int) -> str:
    return f"dictionary{i:02d}.txt"


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as r:
            data = r.read()
        tmp.write_bytes(data)
        tmp.replace(dst)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def ensure_mozc_files(cache_dir: Path, start: int, end: int, raw_base: str, force: bool) -> List[Path]:
    files: List[Path] = []
    for i in range(start, end + 1):
        fn = mozc_filename(i)
        dst = cache_dir / fn
        if force or (not dst.exists()) or dst.stat().st_size == 0:
            url = f"{raw_base}/{fn}"
            print(f"[mozc] download: {url} -> {dst}", file=sys.stderr)
            download_file(url, dst)
        files.append(dst)
    return files


def iter_mozc_entries(
    paths: List[Path],
    require_allowed_surface: bool,
    min_token_len: int,
) -> Iterable[Tuple[str, str]]:
    """
    Yields (reading_hira, surface) from Mozc dictionary files.

    Each line format (TSV):
      読み<TAB>left_id<TAB>right_id<TAB>score<TAB>単語
    """
    for p in paths:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = line.split("\t")
                if len(cols) < 5:
                    continue

                reading_raw = cols[0].strip()
                surface = cols[4].strip()
                if not reading_raw or not surface:
                    continue

                # Mozc reading is usually hiragana already, but normalize defensively.
                reading_hira = kata_to_hira(reading_raw)
                reading_hira = reading_hira.replace(" ", "").replace("\u3000", "")

                if len(reading_hira) < max(1, min_token_len):
                    continue
                if _ALLOWED_READING_RE.match(reading_hira) is None:
                    continue

                if require_allowed_surface and (_ALLOWED_SURFACE_RE.match(surface) is None):
                    continue

                yield (reading_hira, surface)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build seg_boundary jsonl from Mozc dictionary (NO concatenation; each entry is one token)."
    )
    p.add_argument("--out", type=str, required=True, help="output jsonl path (e.g. seg_from_mozc.jsonl)")
    p.add_argument("--split", type=str, default="train", help="id prefix (default: train)")

    # Download / source
    p.add_argument("--mozc_start", type=int, default=0, help="start index (00..)")
    p.add_argument("--mozc_end", type=int, default=9, help="end index (..09)")
    p.add_argument("--mozc_cache_dir", type=str, default=".cache/mozc_dictionary_oss", help="cache dir")
    p.add_argument("--mozc_raw_base", type=str, default=_DEFAULT_MOZC_RAW_BASE, help="raw base url")
    p.add_argument("--mozc_force_download", action="store_true", help="force re-download")

    # Filtering / output control
    p.add_argument("--max_lines", type=int, default=200000, help="stop after N outputs")
    p.add_argument("--dedup_reading", action="store_true", help="deduplicate by reading_hira")
    p.add_argument("--min_token_len", type=int, default=1, help="min reading length (chars)")
    p.add_argument(
        "--require_allowed_surface",
        action="store_true",
        help="require surface to match _ALLOWED_SURFACE_RE",
    )
    p.add_argument(
        "--include_surface",
        action="store_true",
        help="include surface field in output jsonl (for debugging/audit)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cache_dir = Path(args.mozc_cache_dir)
    paths = ensure_mozc_files(
        cache_dir=cache_dir,
        start=args.mozc_start,
        end=args.mozc_end,
        raw_base=args.mozc_raw_base,
        force=args.mozc_force_download,
    )

    n_written = 0
    n_seen = 0
    n_filtered = 0
    n_deduped = 0
    seen_readings: set[str] = set()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fo:
        for (reading_hira, surface) in iter_mozc_entries(
            paths=paths,
            require_allowed_surface=args.require_allowed_surface,
            min_token_len=args.min_token_len,
        ):
            n_seen += 1

            # For seg_boundary task, we need len>=2 to have boundary string length>=1
            if len(reading_hira) < 2:
                n_filtered += 1
                continue

            if args.dedup_reading:
                if reading_hira in seen_readings:
                    n_deduped += 1
                    continue
                seen_readings.add(reading_hira)

            # NO concatenation => one token => no internal boundary.
            boundaries: List[int] = []
            boundary_str = "0" * (len(reading_hira) - 1)

            obj = {
                "id": f"{args.split}:{n_written}",
                "reading_hira": reading_hira,
                "boundaries": boundaries,
                "boundary": boundary_str,
                "analyzer": "mozc",
                "coverage": 1.0,
                "mode": "seg_boundary",
            }
            if args.include_surface:
                obj["surface"] = surface

            fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1

            if n_written >= args.max_lines:
                break

            if n_written % 5000 == 0:
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
