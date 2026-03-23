# safha (صفحة) — Arabic Web Scraper for ML

> Scrape Arabic content. Clean it. Detect dialect. Output training-ready JSONL.

## Why

Building Arabic AI models requires clean Arabic text data. Scraping the web gives you noise — HTML tags, ads, mixed languages, broken encoding. safha handles the full pipeline: scrape, clean, detect dialect, export.

## Install

```bash
pip install safha
```

## Quick Start

```bash
# Scrape a page
safha scrape https://ar.wikipedia.org/wiki/ذكاء_اصطناعي --output data.jsonl

# Crawl an entire sitemap
safha sitemap https://example.com/sitemap.xml

# Clean raw data
safha clean raw_data.jsonl --output clean.jsonl

# Detect dialects
safha detect data.jsonl
```

## Commands

| Command | Description |
|---------|-------------|
| `scrape` | Scrape Arabic content from a URL. Extracts text, strips HTML, outputs structured JSONL. |
| `sitemap` | Crawl all URLs in a sitemap. Parallel fetching with rate limiting. |
| `clean` | Run the full cleaning pipeline on raw scraped data. |
| `detect` | Classify each text by Arabic dialect (MSA, Egyptian, Gulf, Levantine, Moroccan). |
| `stats` | Show dataset statistics — word count, dialect distribution, quality scores. |
| `explain` | Explain what each cleaning step does with before/after examples. |

## Cleaning Pipeline

safha runs text through a multi-step cleaning pipeline:

1. **URL removal** — strip embedded links
2. **Email removal** — strip email addresses
3. **@mention removal** — strip social media handles
4. **Alef normalization** — unify alef variants (أ إ آ ا)
5. **Ya normalization** — unify ya/alef maqsura (ي ى)
6. **Tatweel removal** — strip kashida elongation (ـ)
7. **Tashkeel stripping** — remove diacritics (optional, `--keep-tashkeel` to preserve)
8. **Emoji removal** — strip emoji characters
9. **Sentence splitting** — segment into clean sentences
10. **Quality filtering** — drop too-short, too-long, or low-Arabic-ratio text

## Dialect Detection

Keyword-based classification into 5 dialect categories:

- **MSA** — Modern Standard Arabic (فصحى)
- **Egyptian** — (مصري)
- **Gulf** — (خليجي)
- **Levantine** — (شامي)
- **Moroccan** — (مغربي)

## Output Formats

- **JSONL** — one JSON object per line (default)
- **CSV** — tabular format
- **TXT** — plain text, one document per line
- **HuggingFace datasets** — direct push to Hub

## Supported Providers

No API keys needed. safha scrapes directly from the web using [httpx](https://www.python-httpx.org/) with:

- Automatic encoding detection
- Rate limiting
- Retry with backoff
- Robots.txt respect

---

<p align="center" dir="rtl">مقدمة من <a href="https://x.com/i/communities/2032184341682643429">مجتمع الذكاء الاصطناعي السعودي</a> للعرب أولا وللعالم أجمع</p>

<p align="center">Brought to you by the <a href="https://x.com/i/communities/2032184341682643429">Saudi AI Community</a> — for Arabs first, and the world at large.</p>

## License

MIT License — [Musa the Carpenter](https://github.com/Moshe-ship)

## The Series

[artok](https://github.com/Moshe-ship/artok) · [bidi-guard](https://github.com/Moshe-ship/bidi-guard) · [arabench](https://github.com/Moshe-ship/arabench) · [majal](https://github.com/Moshe-ship/majal) · [khalas](https://github.com/Moshe-ship/khalas) · [safha](https://github.com/Moshe-ship/safha)
