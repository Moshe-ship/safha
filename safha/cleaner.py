"""Arabic text cleaning and processing for ML training data."""
from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CleanConfig:
    """Configuration for the text cleaning pipeline."""

    strip_tashkeel: bool = False
    normalize_alef: bool = True
    normalize_ya: bool = True
    remove_tatweel: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = False
    remove_emojis: bool = True
    min_sentence_length: int = 5
    max_sentence_length: int = 500


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#[\w\u0600-\u06FF]+")

# Emoji ranges (covers most common emoji blocks).
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000200D"             # zero width joiner
    "\U00002B50"             # star
    "]+",
    re.UNICODE,
)

# Arabic tashkeel (diacritics): fathah, dammah, kasrah, sukun, shadda, etc.
_TASHKEEL_RE = re.compile(r"[\u064B-\u065F\u0670]")

# Tatweel / kashida.
_TATWEEL_RE = re.compile(r"\u0640")

# Non-Arabic, non-Latin, non-digit junk (keeps Arabic, Latin, digits,
# common punctuation, and whitespace).
_JUNK_RE = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"
    r"a-zA-Z0-9"
    r"\s.,;:!?\-()\"'\u060C\u061B\u061F\u0022\u0027]"
)

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Arabic letter detection for ratio calculation.
_ARABIC_LETTER_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_LETTER_RE = re.compile(r"\w", re.UNICODE)


# ---------------------------------------------------------------------------
# Cleaning pipeline
# ---------------------------------------------------------------------------


def clean_text(text: str, config: CleanConfig | None = None) -> str:
    """Run the full Arabic text cleaning pipeline.

    Steps (in order):
    1. Remove URLs, emails, @mentions
    2. Optionally remove hashtags, emojis
    3. Normalize alef variants, ya
    4. Remove tatweel
    5. Optionally strip tashkeel
    6. Fix Arabic punctuation spacing
    7. Normalize whitespace
    8. Remove non-Arabic non-Latin non-digit junk
    """
    cfg = config or CleanConfig()

    # 1. URLs / emails / mentions
    if cfg.remove_urls:
        text = _URL_RE.sub("", text)
    if cfg.remove_emails:
        text = _EMAIL_RE.sub("", text)
    if cfg.remove_mentions:
        text = _MENTION_RE.sub("", text)

    # 2. Optional removals
    if cfg.remove_hashtags:
        text = _HASHTAG_RE.sub("", text)
    if cfg.remove_emojis:
        text = _EMOJI_RE.sub("", text)

    # 3. Normalize alef and ya
    if cfg.normalize_alef:
        text = text.replace("\u0623", "\u0627")  # hamza above
        text = text.replace("\u0625", "\u0627")  # hamza below
        text = text.replace("\u0622", "\u0627")  # madda above

    if cfg.normalize_ya:
        text = text.replace("\u0649", "\u064A")  # alef maqsura -> ya

    # 4. Tatweel
    if cfg.remove_tatweel:
        text = _TATWEEL_RE.sub("", text)

    # 5. Tashkeel
    if cfg.strip_tashkeel:
        text = _TASHKEEL_RE.sub("", text)

    # 6. Arabic punctuation spacing: ensure space after Arabic comma,
    #    semicolon, and question mark.
    text = re.sub(r"(\u060C)(\S)", r"\1 \2", text)  # Arabic comma
    text = re.sub(r"(\u061B)(\S)", r"\1 \2", text)  # Arabic semicolon
    text = re.sub(r"(\u061F)(\S)", r"\1 \2", text)  # Arabic question mark

    # 7. Junk removal
    text = _JUNK_RE.sub("", text)

    # 8. Whitespace normalization
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split text on Arabic and Latin sentence boundaries.

    Splits on period, Arabic question mark, exclamation mark, and Arabic
    comma when followed by whitespace.  Applies length filtering based on
    character count.
    """
    # Split on sentence-ending punctuation followed by whitespace or EOL.
    parts = re.split(r"(?<=[.\u061F!\u060C?])\s+", text)

    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if len(part) < 5:
            continue
        if len(part) > 500:
            # Try a secondary split on newlines for very long chunks.
            for sub in part.split("\n"):
                sub = sub.strip()
                if 5 <= len(sub) <= 500:
                    sentences.append(sub)
        else:
            sentences.append(part)

    return sentences


# ---------------------------------------------------------------------------
# Dialect detection
# ---------------------------------------------------------------------------

_DIALECT_MARKERS: dict[str, list[str]] = {
    "Egyptian": [
        "\u0625\u064A\u0647",       # ايه
        "\u062F\u0647",             # ده
        "\u062F\u064A",             # دي
        "\u0627\u0644\u0646\u0647\u0627\u0631\u062F\u0629",  # النهاردة
        "\u0643\u062F\u0647",       # كده
        "\u0639\u0627\u064A\u0632", # عايز
        "\u0627\u0632\u0627\u064A", # ازاي
        "\u062D\u0627\u062C\u0629", # حاجة
        "\u062F\u0644\u0648\u0642\u062A\u064A",  # دلوقتي
    ],
    "Gulf": [
        "\u0634\u0644\u0648\u0646", # شلون
        "\u0648\u0627\u064A\u062F", # وايد
        "\u064A\u0627\u0644\u0644\u0647",  # يالله
        "\u062D\u0642",             # حق
        "\u0634\u062E\u0628\u0627\u0631\u0643",  # شخبارك
        "\u0625\u0646\u0632\u064A\u0646",  # إنزين
        "\u062D\u0628\u064A\u0628\u064A",  # حبيبي
        "\u064A\u0628\u064A\u0644\u0643",  # يبيلك
    ],
    "Levantine": [
        "\u0643\u062A\u064A\u0631", # كتير
        "\u0647\u0644\u0642",       # هلق
        "\u0634\u0648",             # شو
        "\u0643\u064A\u0641\u0643", # كيفك
        "\u0647\u064A\u0643",       # هيك
        "\u0645\u0646\u064A\u062D", # منيح
        "\u0628\u062F\u064A",       # بدي
    ],
    "Moroccan": [
        "\u062F\u064A\u0627\u0644", # ديال
        "\u0628\u063A\u064A\u062A", # بغيت
        "\u0641\u064A\u0646",       # فين
        "\u0648\u0627\u0634",       # واش
        "\u0628\u0632\u0627\u0641", # بزاف
        "\u0645\u0632\u064A\u0627\u0646",  # مزيان
    ],
}


def detect_dialect(text: str) -> str:
    """Classify Arabic text as MSA, Egyptian, Gulf, Levantine, or Moroccan.

    Uses simple keyword matching against known dialect markers.  Returns
    ``"MSA"`` when no strong dialect signal is found.
    """
    scores: dict[str, int] = {dialect: 0 for dialect in _DIALECT_MARKERS}

    for dialect, markers in _DIALECT_MARKERS.items():
        for marker in markers:
            if marker in text:
                scores[dialect] += 1

    best_dialect = max(scores, key=lambda d: scores[d])
    if scores[best_dialect] == 0:
        return "MSA"

    return best_dialect


# ---------------------------------------------------------------------------
# Deduplication and quality filtering
# ---------------------------------------------------------------------------


def _normalize_for_dedup(text: str) -> str:
    """Normalise text for near-duplicate comparison.

    Strips tashkeel, tatweel, normalises alef/ya, lowercases Latin,
    and collapses whitespace.
    """
    t = text.lower()
    t = _TASHKEEL_RE.sub("", t)
    t = _TATWEEL_RE.sub("", t)
    t = t.replace("\u0623", "\u0627")
    t = t.replace("\u0625", "\u0627")
    t = t.replace("\u0622", "\u0627")
    t = t.replace("\u0649", "\u064A")
    t = _MULTI_SPACE.sub(" ", t).strip()
    return t


def deduplicate(texts: list[str]) -> list[str]:
    """Remove exact duplicates and near-duplicates.

    Near-duplicates are detected by comparing a normalised version of
    each text (stripped diacritics, normalised alef/ya, lowercased).
    """
    seen: set[str] = set()
    result: list[str] = []

    for text in texts:
        key = _normalize_for_dedup(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)

    return result


def filter_quality(
    texts: list[str],
    min_words: int = 20,
    min_arabic_ratio: float = 0.3,
) -> list[str]:
    """Remove low-quality texts that are too short or lack Arabic content."""
    result: list[str] = []

    for text in texts:
        words = text.split()
        if len(words) < min_words:
            continue

        letters = _LETTER_RE.findall(text)
        if not letters:
            continue

        arabic_count = sum(1 for ch in letters if _ARABIC_LETTER_RE.match(ch))
        ratio = arabic_count / len(letters)
        if ratio < min_arabic_ratio:
            continue

        result.append(text)

    return result
