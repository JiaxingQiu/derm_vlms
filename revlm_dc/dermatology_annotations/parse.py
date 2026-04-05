"""Extract diagnoses and descriptions from free-text VLM responses.

Consolidates the parsing logic originally developed in
res_eng/interface/notebook.ipynb so it can be shared by the Django
annotation interface and the automated evaluation pipeline.
"""

import re

DIAG_PATTERNS = [
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
]


def split_response(text):
    """Split a VLM response into (description, diagnoses).

    Tries each DIAG_PATTERN in order. On the first match, everything
    before is description and the match + everything after is diagnoses.
    """
    if not text:
        return ("", "")
    for pat in DIAG_PATTERNS:
        m = pat.search(text)
        if m:
            return (text[: m.start()].strip(), text[m.start() :].strip())
    return (text.strip(), "")


def clean_markdown(text):
    """Strip markdown formatting, horizontal rules, and excess whitespace."""
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


def split_sentences(text):
    """Split text into sentences on period boundaries.

    Keeps decimals like '0.5 cm' and common abbreviations intact.
    """
    if not text:
        return []
    parts = re.findall(r"[^.]*\.", text)
    if not parts:
        return [text.strip()] if text.strip() else []

    sentences = []
    buf = ""
    for part in parts:
        buf += part
        stripped = buf.strip()
        if not stripped:
            continue
        if re.search(r"\d\.$", stripped) and re.search(
            r"\d\.\d", text[text.find(stripped) : text.find(stripped) + len(stripped) + 2]
        ):
            continue
        if re.search(r"\b(?:Dr|Mr|Mrs|Ms|vs|etc|approx|e\.g|i\.e|Fig|No)\.$", stripped):
            continue
        sentences.append(stripped)
        buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


def parse_diagnoses(diag_text):
    """Extract individual numbered diagnoses from the diagnosis section."""
    if not diag_text:
        return []

    pat_dot = re.compile(r"(?:^|\n)\s*\d+\.\s+", re.MULTILINE)
    matches = list(pat_dot.finditer(diag_text))

    if not matches:
        pat_paren = re.compile(r"\d+\)\s*")
        matches = list(pat_paren.finditer(diag_text))

    if not matches:
        return []

    items = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(diag_text)
        text = diag_text[start:end].strip()
        text = re.sub(r"[,;]\s*$", "", text)
        text = re.sub(r"\s+and\s*$", "", text, flags=re.IGNORECASE)
        if text:
            items.append(text)
    return items


def _extract_name(raw_item):
    """Extract the short diagnosis name from a parsed item.

    Strips trailing explanations (after dashes, colons, etc.) and
    parenthetical notes while keeping the core diagnosis name.
    """
    name = re.split(r"\s*[-–—]\s+", raw_item, maxsplit=1)[0]
    name = re.split(r"\s*:\s+", name, maxsplit=1)[0]
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    name = re.split(
        r",\s+(?:given|due|which|although|though|as|since|because)\b",
        name, maxsplit=1,
    )[0]
    name = re.split(r"\s+with\s+(?:a\s+)?(?:more|some|secondary|possible)", name, maxsplit=1)[0]
    return name.strip().rstrip(".,;")


def extract_top3(text, max_items=3):
    """Extract up to 3 diagnosis names from a full VLM response.

    Returns a list of cleaned diagnosis name strings.
    """
    _, diag_raw = split_response(text)
    diag_clean = clean_markdown(diag_raw)
    items = parse_diagnoses(diag_clean)
    return [_extract_name(item) for item in items[:max_items]]


def parse_response(text):
    """Parse a full VLM response into descriptions and diagnoses lists.

    Returns (descriptions: list[str], diagnoses: list[str]) ready for
    the Django annotation interface.
    """
    desc_raw, diag_raw = split_response(text)
    desc_clean = clean_markdown(desc_raw)
    diag_clean = clean_markdown(diag_raw)
    descriptions = split_sentences(desc_clean)
    diagnoses = parse_diagnoses(diag_clean)
    return descriptions, diagnoses
