"""Arabic web scraping engine."""
from __future__ import annotations

import html as html_module
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import httpx


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ScrapedPage:
    """A single scraped web page with extracted Arabic content."""

    url: str
    title: str
    raw_html: str
    clean_text: str
    arabic_text: str
    word_count: int
    arabic_ratio: float
    dialect: str | None
    scraped_at: str


@dataclass
class ScrapeConfig:
    """Configuration for the scraping engine."""

    max_pages: int = 50
    delay: float = 1.0
    timeout: float = 15.0
    min_words: int = 20
    min_arabic_ratio: float = 0.3
    user_agent: str = "Mozilla/5.0 (compatible; safha/0.1; +https://github.com/Moshe-ship/safha)"


# ---------------------------------------------------------------------------
# Arabic detection helpers
# ---------------------------------------------------------------------------

_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_LETTER_RE = re.compile(r"\w", re.UNICODE)


# ---------------------------------------------------------------------------
# HTML cleaning (regex-based, no bs4 dependency)
# ---------------------------------------------------------------------------

# Block-level elements whose content should be stripped entirely.
_STRIP_BLOCKS = re.compile(
    r"<\s*(script|style|nav|header|footer|aside|noscript)[^>]*>.*?</\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_page(url: str, config: ScrapeConfig | None = None) -> str | None:
    """Fetch a URL and return its HTML, or *None* on failure.

    Respects *config.delay* by sleeping before the request and uses
    *config.timeout* / *config.user_agent* for the HTTP call.
    """
    cfg = config or ScrapeConfig()

    if cfg.delay > 0:
        time.sleep(cfg.delay)

    try:
        resp = httpx.get(
            url,
            timeout=cfg.timeout,
            headers={"User-Agent": cfg.user_agent},
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        return None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text(html: str) -> str:
    """Extract visible text from raw HTML using regex.

    Strips <script>, <style>, <nav>, <header>, <footer>, and <aside>
    blocks, removes all remaining HTML tags, decodes HTML entities, and
    normalises whitespace.
    """
    text = _STRIP_BLOCKS.sub("", html)
    text = _HTML_TAG.sub(" ", text)
    text = html_module.unescape(text)
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def extract_title(html: str) -> str:
    """Return the content of the first <title> tag, or an empty string."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return html_module.unescape(match.group(1)).strip()
    return ""


# ---------------------------------------------------------------------------
# Arabic filtering and measurement
# ---------------------------------------------------------------------------


def filter_arabic(text: str) -> str:
    """Keep only lines / sentences that contain at least one Arabic character."""
    lines = text.split("\n")
    arabic_lines = [line for line in lines if _ARABIC_RE.search(line)]
    return "\n".join(arabic_lines).strip()


def count_arabic_ratio(text: str) -> float:
    """Return the ratio of Arabic characters to total letter characters.

    Returns 0.0 when the text has no letter characters at all.
    """
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
    arabic_count = sum(1 for ch in letters if _ARABIC_RE.match(ch))
    return arabic_count / len(letters)


# ---------------------------------------------------------------------------
# Scraping pipeline
# ---------------------------------------------------------------------------


def scrape_url(url: str, config: ScrapeConfig | None = None) -> ScrapedPage | None:
    """Fetch, extract, filter, and build a *ScrapedPage* for a single URL.

    Returns *None* if the page fails to fetch or does not meet the
    minimum word-count / Arabic-ratio thresholds.
    """
    cfg = config or ScrapeConfig()

    raw_html = fetch_page(url, cfg)
    if raw_html is None:
        return None

    title = extract_title(raw_html)
    clean_text = extract_text(raw_html)
    arabic_text = filter_arabic(clean_text)

    word_count = len(arabic_text.split())
    if word_count < cfg.min_words:
        return None

    arabic_ratio = count_arabic_ratio(arabic_text)
    if arabic_ratio < cfg.min_arabic_ratio:
        return None

    return ScrapedPage(
        url=url,
        title=title,
        raw_html=raw_html,
        clean_text=clean_text,
        arabic_text=arabic_text,
        word_count=word_count,
        arabic_ratio=arabic_ratio,
        dialect=None,  # filled later by cleaner.detect_dialect
        scraped_at=datetime.now(timezone.utc).isoformat(),
    )


def scrape_urls(
    urls: list[str],
    config: ScrapeConfig | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[ScrapedPage]:
    """Scrape multiple URLs with a delay between requests.

    *progress_callback*, if provided, is called as
    ``callback(current_index, total, url)`` before each request.
    """
    cfg = config or ScrapeConfig()
    pages: list[ScrapedPage] = []

    for idx, url in enumerate(urls[: cfg.max_pages]):
        if progress_callback:
            progress_callback(idx, len(urls), url)

        page = scrape_url(url, cfg)
        if page is not None:
            pages.append(page)

    return pages


# ---------------------------------------------------------------------------
# Sitemap support
# ---------------------------------------------------------------------------


def scrape_sitemap(
    sitemap_url: str,
    config: ScrapeConfig | None = None,
) -> list[ScrapedPage]:
    """Parse a sitemap.xml, extract <loc> URLs, and scrape them all."""
    cfg = config or ScrapeConfig()

    xml = fetch_page(sitemap_url, cfg)
    if xml is None:
        return []

    urls = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, re.IGNORECASE)
    return scrape_urls(urls, cfg)


# ---------------------------------------------------------------------------
# Search (not implemented)
# ---------------------------------------------------------------------------


def scrape_search(
    query: str,
    num_results: int = 10,
    config: ScrapeConfig | None = None,
) -> list[ScrapedPage]:
    """Search-based scraping -- NOT implemented.

    We intentionally do not scrape Google or any search engine.
    Use ``scrape_urls`` or ``scrape_sitemap`` with known URLs instead.
    """
    return []
