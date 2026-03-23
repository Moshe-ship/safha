"""Rich terminal display for safha results."""

from __future__ import annotations

import json
from collections import Counter
from urllib.parse import urlparse

from rich.console import Console
from rich.table import Table

console = Console()

BRANDING = "[bold magenta]safha[/bold magenta] [dim]- Arabic Web Scraper for ML[/dim]"


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def display_results(pages: list[dict]) -> None:
    """Show a summary table after scraping."""
    if not pages:
        console.print("[yellow]No pages to display.[/yellow]")
        return

    total_pages = len(pages)
    total_words = sum(p.get("word_count", 0) for p in pages)
    ratios = [p.get("arabic_ratio", 0.0) for p in pages]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0

    console.print()
    console.print(BRANDING)
    console.print()
    console.print(
        f"[bold]Pages:[/bold] {total_pages}  "
        f"[bold]Words:[/bold] {total_words:,}  "
        f"[bold]Avg Arabic:[/bold] {avg_ratio:.0%}"
    )
    console.print()

    table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        padding=(0, 1),
    )

    table.add_column("URL", style="bold white", max_width=50)
    table.add_column("Title", style="dim white", max_width=30)
    table.add_column("Words", justify="right", style="white")
    table.add_column("Arabic %", justify="right")
    table.add_column("Dialect", style="yellow")

    for p in pages:
        ratio = p.get("arabic_ratio", 0.0)
        if ratio >= 0.70:
            ratio_style = "green"
        elif ratio >= 0.40:
            ratio_style = "yellow"
        else:
            ratio_style = "red"

        table.add_row(
            p.get("url", ""),
            p.get("title", ""),
            str(p.get("word_count", 0)),
            f"[{ratio_style}]{ratio:.0%}[/{ratio_style}]",
            p.get("dialect", ""),
        )

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


def display_progress(current: int, total: int, url: str) -> None:
    """Print a progress line during scraping."""
    console.print(
        f"[dim][{current}/{total}][/dim] {url}",
        highlight=False,
    )


# ---------------------------------------------------------------------------
# Export confirmation
# ---------------------------------------------------------------------------


def display_export(count: int, format: str, path: str) -> None:
    """Confirm an export operation."""
    console.print(
        f"[green]Exported {count} rows to {path} ({format})[/green]"
    )


# ---------------------------------------------------------------------------
# Detailed stats
# ---------------------------------------------------------------------------


def display_stats(pages: list[dict]) -> None:
    """Show detailed statistics: dialect distribution, word count buckets, top domains."""
    if not pages:
        console.print("[yellow]No data to analyze.[/yellow]")
        return

    console.print()
    console.print(BRANDING)
    console.print()

    # Dialect distribution
    dialects = Counter(p.get("dialect", "unknown") for p in pages)
    console.print("[bold]Dialect distribution:[/bold]")
    console.print()
    for dialect, count in dialects.most_common():
        pct = count / len(pages)
        console.print(f"  {dialect:<16} {count:>5}  ({pct:.0%})")
    console.print()

    # Word count histogram buckets
    word_counts = [p.get("word_count", 0) for p in pages]
    buckets = [
        ("0-50", 0, 50),
        ("51-200", 51, 200),
        ("201-500", 201, 500),
        ("501-1000", 501, 1000),
        ("1000+", 1001, float("inf")),
    ]

    console.print("[bold]Word count distribution:[/bold]")
    console.print()
    for label, lo, hi in buckets:
        count = sum(1 for w in word_counts if lo <= w <= hi)
        console.print(f"  {label:<12} {count:>5}")
    console.print()

    # Top domains
    domains = Counter(
        urlparse(p.get("url", "")).netloc for p in pages
    )
    console.print("[bold]Top domains:[/bold]")
    console.print()
    for domain, count in domains.most_common(10):
        console.print(f"  {domain:<30} {count:>5}")
    console.print()


# ---------------------------------------------------------------------------
# Config status
# ---------------------------------------------------------------------------


def display_config_status() -> None:
    """Show the current scrape configuration."""
    console.print()
    console.print(BRANDING)
    console.print()
    console.print("[bold]Current configuration:[/bold]")
    console.print()
    console.print("  Output format:   jsonl")
    console.print("  Max pages:       50")
    console.print("  Delay:           1.0s")
    console.print("  Min words:       20")
    console.print("  Min Arabic:      30%")
    console.print("  Strip tashkeel:  no")
    console.print("  Keep hashtags:   no")
    console.print()


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


def display_explain() -> None:
    """Explain what safha does, the cleaning pipeline, dialect detection, and output formats."""
    console.print()
    console.print(BRANDING)
    console.print()

    console.print("[bold]What is safha?[/bold]")
    console.print()
    console.print(
        "  safha scrapes Arabic web pages and prepares the text for machine learning.\n"
        "  It handles the full pipeline: fetching, cleaning, dialect detection,\n"
        "  and exporting to ML-ready formats.\n"
    )

    console.print("[bold]Cleaning pipeline:[/bold]")
    console.print()
    console.print("  1. Fetch HTML via httpx with configurable delay")
    console.print("  2. Extract main content (strip nav, footer, ads, scripts)")
    console.print("  3. Normalize Unicode (NFKC)")
    console.print("  4. Remove URLs, emails, and non-Arabic noise")
    console.print("  5. Optionally strip tashkeel (diacritics)")
    console.print("  6. Optionally remove hashtags")
    console.print("  7. Filter by minimum word count and Arabic ratio")
    console.print()

    console.print("[bold]Dialect detection:[/bold]")
    console.print()
    console.print(
        "  safha classifies each page into one of: MSA, Egyptian, Gulf,\n"
        "  Levantine, Maghrebi, or Mixed, using lexical heuristics.\n"
    )

    console.print("[bold]Output formats:[/bold]")
    console.print()
    console.print("  jsonl  - One JSON object per line (default)")
    console.print("  csv    - Comma-separated values")
    console.print("  txt    - Plain text, one document per line")
    console.print("  hf     - Hugging Face Dataset (Arrow)")
    console.print()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def display_json(data: object) -> None:
    """Print data as formatted JSON."""
    console.print(json.dumps(data, ensure_ascii=False, indent=2))
