"""CLI entry point for safha."""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from safha.display import (
    console,
    display_explain,
    display_export,
    display_json,
    display_results,
    display_stats,
)


# ── Shared arguments ──────────────────────────────────────────────


def _common_scrape_args(parser: argparse.ArgumentParser) -> None:
    """Add flags shared by scrape and sitemap subcommands."""
    parser.add_argument(
        "--output",
        default="output.jsonl",
        metavar="FILE",
        help="Output file path (default: output.jsonl)",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv", "txt", "hf"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        metavar="N",
        help="Maximum pages to scrape (default: 50)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Delay between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=20,
        metavar="N",
        help="Minimum words per page (default: 20)",
    )
    parser.add_argument(
        "--min-arabic",
        type=float,
        default=0.3,
        metavar="RATIO",
        help="Minimum Arabic character ratio (default: 0.3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON instead of table",
    )
    parser.add_argument(
        "--strip-tashkeel",
        action="store_true",
        default=False,
        help="Strip diacritics (tashkeel) from text",
    )
    parser.add_argument(
        "--keep-hashtags",
        action="store_true",
        default=False,
        help="Keep hashtags instead of removing them",
    )


# ── Subcommand handlers ──────────────────────────────────────────


def _cmd_scrape(args: argparse.Namespace) -> int:
    """Run the scrape subcommand."""
    from safha.scraper import scrape_urls, ScrapeConfig
    from safha.cleaner import CleanConfig, clean_text, detect_dialect
    from safha.exporter import export

    config = ScrapeConfig(
        max_pages=args.max_pages,
        delay=args.delay,
        min_words=args.min_words,
        min_arabic_ratio=args.min_arabic,
    )

    with console.status("[bold cyan]Scraping...", spinner="dots"):
        pages = scrape_urls(args.urls, config)

    # Clean and detect dialect
    clean_cfg = CleanConfig(
        strip_tashkeel=getattr(args, "strip_tashkeel", False),
        remove_hashtags=not getattr(args, "keep_hashtags", False),
    )
    for page in pages:
        page.clean_text = clean_text(page.clean_text, clean_cfg)
        page.dialect = detect_dialect(page.clean_text)

    page_dicts = [_page_to_dict(p) for p in pages]
    if args.json:
        display_json(page_dicts)
    else:
        display_results(page_dicts)

    # Export
    if pages:
        fmt = getattr(args, "format", "jsonl")
        output = getattr(args, "output", "output.jsonl")
        count = export(pages, output, fmt)
        display_export(count, fmt, output)

    return 0


def _cmd_sitemap(args: argparse.Namespace) -> int:
    """Run the sitemap subcommand."""
    from safha.scraper import scrape_sitemap, ScrapeConfig
    from safha.cleaner import CleanConfig, clean_text, detect_dialect
    from safha.exporter import export

    config = ScrapeConfig(
        max_pages=args.max_pages,
        delay=args.delay,
        min_words=args.min_words,
        min_arabic_ratio=args.min_arabic,
    )

    with console.status(f"[bold cyan]Fetching sitemap {args.url}...", spinner="dots"):
        pages = scrape_sitemap(args.url, config)

    clean_cfg = CleanConfig(
        strip_tashkeel=getattr(args, "strip_tashkeel", False),
        remove_hashtags=not getattr(args, "keep_hashtags", False),
    )
    for page in pages:
        page.clean_text = clean_text(page.clean_text, clean_cfg)
        page.dialect = detect_dialect(page.clean_text)

    page_dicts = [_page_to_dict(p) for p in pages]
    if args.json:
        display_json(page_dicts)
    else:
        display_results(page_dicts)

    if pages:
        fmt = getattr(args, "format", "jsonl")
        output = getattr(args, "output", "output.jsonl")
        count = export(pages, output, fmt)
        display_export(count, fmt, output)

    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    """Run the clean subcommand — clean an existing text file."""
    from safha.cleaner import CleanConfig, clean_text
    import json

    clean_cfg = CleanConfig(
        strip_tashkeel=getattr(args, "strip_tashkeel", False),
        remove_hashtags=not getattr(args, "keep_hashtags", False),
    )

    input_path = args.file
    output_path = getattr(args, "output", None) or input_path.rsplit(".", 1)[0] + ".cleaned.jsonl"

    rows = []
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {"text": line}
            for key, val in row.items():
                if isinstance(val, str) and len(val) > 10:
                    row[key] = clean_text(val, clean_cfg)
            rows.append(row)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    console.print(f"[green]Cleaned {len(rows)} rows → {output_path}[/green]")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    """Run the detect subcommand — detect dialect of each row."""
    from safha.cleaner import detect_dialect
    import json

    input_path = args.file
    output_path = getattr(args, "output", None) or input_path.rsplit(".", 1)[0] + ".dialects.jsonl"

    rows = []
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {"text": line}
            # Find text field and detect dialect
            text = row.get("text", row.get("arabic", row.get("content", "")))
            row["dialect"] = detect_dialect(text)
            rows.append(row)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    dialects = Counter(r.get("dialect", "unknown") for r in rows)
    console.print("\n[bold magenta]safha[/bold magenta] [dim]- Dialect Detection[/dim]\n")
    for dialect, count in dialects.most_common():
        console.print(f"  {dialect:12s}  {count}")
    console.print(f"\n[green]Wrote {len(rows)} rows → {output_path}[/green]")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """Run the stats subcommand — show dataset statistics."""
    import json

    rows = []
    with open(args.file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {"text": line}
            rows.append(row)

    # Build fake pages for display_stats
    from safha.scraper import ScrapedPage
    from safha.scraper import count_arabic_ratio
    pages = []
    for row in rows:
        text = row.get("text", row.get("arabic", row.get("content", "")))
        pages.append(ScrapedPage(
            url=row.get("url", ""),
            title="",
            raw_html="",
            clean_text=text,
            arabic_text=text,
            word_count=len(text.split()),
            arabic_ratio=count_arabic_ratio(text),
            dialect=row.get("dialect"),
            scraped_at="",
        ))

    display_stats([_page_to_dict(p) for p in pages])
    return 0


def _page_to_dict(page) -> dict:
    """Convert ScrapedPage to dict for JSON output."""
    return {
        "url": page.url,
        "title": page.title,
        "text": page.clean_text,
        "word_count": page.word_count,
        "arabic_ratio": round(page.arabic_ratio, 3),
        "dialect": page.dialect,
    }


def _cmd_explain(_args: argparse.Namespace) -> int:
    """Run the explain subcommand — show pipeline info."""
    display_explain()
    return 0


# ── Argument parser construction ──────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="safha",
        description="Arabic Web Scraper for ML — fetch, clean, and export Arabic text.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── scrape ──
    scrape_parser = subparsers.add_parser(
        "scrape", help="Scrape one or more URLs (default)"
    )
    scrape_parser.add_argument(
        "urls",
        nargs="+",
        metavar="URL",
        help="URLs to scrape",
    )
    _common_scrape_args(scrape_parser)

    # ── sitemap ──
    sitemap_parser = subparsers.add_parser(
        "sitemap", help="Scrape all pages from a sitemap.xml"
    )
    sitemap_parser.add_argument(
        "url",
        metavar="URL",
        help="Sitemap URL to fetch",
    )
    _common_scrape_args(sitemap_parser)

    # ── clean ──
    clean_parser = subparsers.add_parser(
        "clean", help="Clean an existing text file (JSONL/CSV/TXT)"
    )
    clean_parser.add_argument(
        "file",
        metavar="FILE",
        help="Input file to clean",
    )
    clean_parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output file path (default: overwrite input)",
    )
    clean_parser.add_argument(
        "--strip-tashkeel",
        action="store_true",
        default=False,
        help="Strip diacritics (tashkeel) from text",
    )
    clean_parser.add_argument(
        "--keep-hashtags",
        action="store_true",
        default=False,
        help="Keep hashtags instead of removing them",
    )

    # ── detect ──
    detect_parser = subparsers.add_parser(
        "detect", help="Detect dialect of each row in a file"
    )
    detect_parser.add_argument(
        "file",
        metavar="FILE",
        help="Input file to analyze",
    )
    detect_parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output file with dialect column added",
    )

    # ── stats ──
    stats_parser = subparsers.add_parser(
        "stats", help="Show statistics of a scraped/cleaned dataset"
    )
    stats_parser.add_argument(
        "file",
        metavar="FILE",
        help="Dataset file to analyze",
    )

    # ── explain ──
    subparsers.add_parser(
        "explain", help="Explain the safha pipeline"
    )

    return parser


def _get_version() -> str:
    """Return the package version string."""
    from safha import __version__

    return __version__


# ── Entry point ───────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> NoReturn:
    """Main entry point for the safha CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Default to scrape when no subcommand is given.
    if args.command is None:
        args = parser.parse_args(["scrape", *(argv or sys.argv[1:])])

    dispatch = {
        "scrape": _cmd_scrape,
        "sitemap": _cmd_sitemap,
        "clean": _cmd_clean,
        "detect": _cmd_detect,
        "stats": _cmd_stats,
        "explain": _cmd_explain,
    }

    try:
        handler = dispatch[args.command]
        code = handler(args)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        code = 130
    except BrokenPipeError:
        # Silently handle piping to head/less.
        code = 0

    sys.exit(code)
