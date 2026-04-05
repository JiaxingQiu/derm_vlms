"""Extract top-3 diagnoses from free-text VLM responses.

Self-contained parsing for the eval pipeline.
"""

import re

# ---------------------------------------------------------------------------
# 1. split_response: find the boundary between description and diagnosis
# ---------------------------------------------------------------------------

_DIAG_PATTERNS = [
    re.compile(
        r'^\s*\*{0,2}(?:Top\s+\d+\s+)?Differential\s+Diagnos[ei]s'
        r'(?:\s*\(Top\s*\d+\))?[:\s*]*\*{0,2}\s*$',
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r'^(?:#*\s*)?(?:\*{0,2})Top\s+(?:\d+\s+)?differential\s+diagnos[ei]s[:\s]*(?:\*{0,2})',
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r'^(?:#*\s*)?(?:\*{0,2})Top\s+\d+\s+diagnos[ei]s[:\s]*(?:\*{0,2})',
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r'(?:Based on|Given)\s+.{0,80}?(?:differential|top\s+\d+)\s+diagnos[ei]s\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'(?:the|my)\s+differential\s+(?:diagnos[ei]s\s+)?(?:would\s+)?include[s]?\s*:',
        re.IGNORECASE,
    ),
    re.compile(
        r'(?:the\s+)?(?:top\s+)?differentials?\s+(?:are|include)[s]?\s*:',
        re.IGNORECASE,
    ),
    re.compile(
        r'(?:the\s+)?differential\s+diagnos[ei]s\s+include[s]?\s*:',
        re.IGNORECASE,
    ),
    # Broader: "differential diagnosis would include ..."  (no colon)
    re.compile(
        r'(?:the\s+)?differential\s+(?:diagnos[ei]s\s+)?(?:would\s+)?include[s]?',
        re.IGNORECASE,
    ),
    # "The differential would include ..."
    re.compile(
        r'(?:the\s+)?differential\s+would\s+include',
        re.IGNORECASE,
    ),
    # "the differential diagnosis includes ..."
    re.compile(
        r'(?:the\s+)?differential\s+(?:diagnos[ei]s\s+)?includes?\b',
        re.IGNORECASE,
    ),
    # "would be:" patterns — "diagnoses would be: 1) ..."
    re.compile(
        r'(?:the\s+)?(?:top\s+\d?\s*)?(?:differential\s+)?diagnos[ei]s\s+would\s+be\s*:?',
        re.IGNORECASE,
    ),
]


def split_response(text):
    if not text:
        return ("", "")
    for pat in _DIAG_PATTERNS:
        m = pat.search(text)
        if m:
            return (text[: m.start()].strip(), text[m.start() :].strip())
    return (text.strip(), "")


# ---------------------------------------------------------------------------
# 2. clean_markdown
# ---------------------------------------------------------------------------

def clean_markdown(text):
    if not text:
        return ""
    t = text
    t = re.sub(r"\*{3,}", "", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.DOTALL)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", t, flags=re.DOTALL)
    t = t.replace("*", "")
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"^[-_]{3,}\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---------------------------------------------------------------------------
# 3. parse_diagnoses: extract individual diagnosis items
# ---------------------------------------------------------------------------

def _parse_numbered(diag_text):
    """Try numbered patterns:  1. X  /  1) X  /  inline 1. X, 2. Y"""
    # Multi-line numbered: "1. X\n2. Y" (start of line)
    pat_dot_ml = re.compile(r"(?:^|\n)\s*\d+\.\s+", re.MULTILINE)
    matches = list(pat_dot_ml.finditer(diag_text))
    if len(matches) >= 2:
        return _items_from_matches(matches, diag_text)

    # Inline "1. X, 2. Y, 3. Z" (anywhere)
    pat_dot_inline = re.compile(r"\d+\.\s+")
    matches = list(pat_dot_inline.finditer(diag_text))
    if len(matches) >= 2:
        return _items_from_matches(matches, diag_text)

    # Paren style: 1) X  2) Y
    pat_paren = re.compile(r"\d+\)\s*")
    matches = list(pat_paren.finditer(diag_text))
    if matches:
        return _items_from_matches(matches, diag_text)

    return []


def _items_from_matches(matches, text):
    items = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item = text[start:end].strip()
        item = re.sub(r"[,;]\s*$", "", item)
        item = re.sub(r"\s+and\s*$", "", item, flags=re.IGNORECASE)
        if item:
            items.append(item)
    return items


def _parse_comma_list(diag_text):
    """Fallback: parse 'include X, Y, and/or Z' style comma-separated lists."""
    m = re.search(
        r'include[s]?\s+(.+?)(?:\.|$)',
        diag_text, re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return []

    body = m.group(1).strip()
    # Split on ", and ", ", or ", "and ", commas
    parts = re.split(r'\s*,\s*(?:and|or)\s+|\s*,\s+|\s+and\s+|\s+or\s+', body)
    # Clean leading articles
    cleaned = []
    for p in parts:
        p = re.sub(r'^(?:an?\s+)', '', p.strip(), flags=re.IGNORECASE)
        p = p.strip().rstrip(".,;")
        if p and len(p) > 2:
            cleaned.append(p)
    return cleaned


def parse_diagnoses(diag_text):
    if not diag_text:
        return []
    items = _parse_numbered(diag_text)
    if items:
        return items
    return _parse_comma_list(diag_text)


# ---------------------------------------------------------------------------
# 4. _extract_name: clean a single diagnosis item to its core name
# ---------------------------------------------------------------------------

def _extract_name(raw_item):
    name = re.split(r"\s*[-–—]\s+", raw_item, maxsplit=1)[0]
    name = re.split(r"\s*:\s+", name, maxsplit=1)[0]
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    name = re.split(
        r",\s+(?:given|due|which|although|though|as|since|because)\b",
        name, maxsplit=1,
    )[0]
    name = re.split(
        r"\s+with\s+(?:a\s+)?(?:more|some|secondary|possible)",
        name, maxsplit=1,
    )[0]
    return name.strip().rstrip(".,;")


# ---------------------------------------------------------------------------
# 5. extract_top3: main entry point
# ---------------------------------------------------------------------------

_INLINE_DIAG_PATTERNS = [
    # "consistent with a [diagnosis]"
    re.compile(
        r'consistent\s+with\s+(?:an?\s+)?(.+?)(?:\.|,\s+(?:but|however|and\s+(?:the|there|is))|$)',
        re.IGNORECASE,
    ),
    # "suggestive of [diagnosis]"
    re.compile(
        r'suggestive\s+of\s+(?:an?\s+)?(.+?)(?:\.|,\s+(?:but|however)|$)',
        re.IGNORECASE,
    ),
    # "likely [a] [diagnosis]"
    re.compile(
        r'(?:most\s+)?likely\s+(?:represents?\s+)?(?:an?\s+)?(.+?)(?:\.|,\s+(?:but|however)|$)',
        re.IGNORECASE,
    ),
]


def _extract_inline_single(text):
    """Last-resort: pull a single diagnosis from 'consistent with X' etc."""
    for pat in _INLINE_DIAG_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = m.group(1).strip().rstrip(".,;")
            if len(candidate) > 3 and len(candidate) < 100:
                return [candidate]
    return []


def extract_top3(text, max_items=3):
    """Extract up to 3 diagnosis names from a full VLM response."""
    _, diag_raw = split_response(text)
    diag_clean = clean_markdown(diag_raw)
    items = parse_diagnoses(diag_clean)
    if items:
        return [_extract_name(item) for item in items[:max_items]]
    # Last resort: look for single inline diagnosis in full text
    return _extract_inline_single(text)
