"""Export scraped data to ML training formats."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scraper import ScrapedPage


# ---------------------------------------------------------------------------
# Default fields
# ---------------------------------------------------------------------------

DEFAULT_FIELDS: dict[str, str] = {
    "text": "arabic_text",
    "url": "url",
    "dialect": "dialect",
    "word_count": "word_count",
}


# ---------------------------------------------------------------------------
# JSONL export
# ---------------------------------------------------------------------------


def to_jsonl(
    pages: list[ScrapedPage],
    output_path: str | Path,
    fields: dict[str, str] | None = None,
) -> int:
    """Write pages to a JSONL file with configurable field mapping.

    *fields* maps output-key -> ScrapedPage-attribute.  For example,
    ``{"text": "arabic_text", "url": "url"}`` writes each row as
    ``{"text": "...", "url": "..."}``.

    Returns the number of rows written.
    """
    field_map = fields or DEFAULT_FIELDS
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with path.open("w", encoding="utf-8") as fh:
        for page in pages:
            page_dict = asdict(page)
            row = {}
            for out_key, attr in field_map.items():
                row[out_key] = page_dict.get(attr, "")
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

    return rows


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def to_csv(
    pages: list[ScrapedPage],
    output_path: str | Path,
    fields: dict[str, str] | None = None,
) -> int:
    """Write pages to a CSV file with configurable field mapping.

    Returns the number of rows written.
    """
    field_map = fields or DEFAULT_FIELDS
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(field_map.keys()))
        writer.writeheader()

        for page in pages:
            page_dict = asdict(page)
            row = {}
            for out_key, attr in field_map.items():
                row[out_key] = page_dict.get(attr, "")
            writer.writerow(row)
            rows += 1

    return rows


# ---------------------------------------------------------------------------
# Plain text export
# ---------------------------------------------------------------------------


def to_txt(
    pages: list[ScrapedPage],
    output_path: str | Path,
) -> int:
    """Write pages as plain text, one document per line.

    Returns the number of rows written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with path.open("w", encoding="utf-8") as fh:
        for page in pages:
            # Collapse newlines within a document to spaces for one-line format.
            line = page.arabic_text.replace("\n", " ").strip()
            if line:
                fh.write(line + "\n")
                rows += 1

    return rows


# ---------------------------------------------------------------------------
# HuggingFace datasets format
# ---------------------------------------------------------------------------


def to_huggingface(
    pages: list[ScrapedPage],
    output_path: str | Path,
) -> int:
    """Write pages in HuggingFace datasets format (JSONL with "text" only).

    Returns the number of rows written.
    """
    return to_jsonl(pages, output_path, fields={"text": "arabic_text"})


# ---------------------------------------------------------------------------
# Format dispatcher
# ---------------------------------------------------------------------------

_FORMATS = {
    "jsonl": to_jsonl,
    "csv": to_csv,
    "txt": to_txt,
    "huggingface": to_huggingface,
    "hf": to_huggingface,
}


def export(
    pages: list[ScrapedPage],
    output_path: str | Path,
    fmt: str = "jsonl",
    fields: dict[str, str] | None = None,
) -> int:
    """Export pages in the requested format.

    Supported formats: ``jsonl``, ``csv``, ``txt``, ``huggingface`` / ``hf``.
    Returns the number of rows written.

    Raises *ValueError* for unknown formats.
    """
    fmt_lower = fmt.lower()
    handler = _FORMATS.get(fmt_lower)
    if handler is None:
        supported = ", ".join(sorted(_FORMATS.keys()))
        raise ValueError(f"Unknown format {fmt!r}. Supported: {supported}")

    # txt and huggingface don't accept fields
    if fmt_lower in ("txt", "huggingface", "hf"):
        return handler(pages, output_path)

    return handler(pages, output_path, fields=fields)
