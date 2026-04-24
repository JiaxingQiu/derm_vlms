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


_REASONING_LABEL = re.compile(
    r"^\s*(?:reasoning|rationale|explanation|why|justification)\s*[:\-–—]\s*",
    re.IGNORECASE,
)
_LEADING_BULLET = re.compile(r"^\s*(?:[*•\-–—\u2022]\s*)+")
_NAME_REASON_SPLIT = re.compile(r"\s*[:\-–—]\s+")
# Reason-format responses usually begin with "1. ..." (possibly after a short
# preamble). If we detect that shape anywhere near the top we should parse the
# whole text as the diagnosis list rather than relying on split_response.
_REASON_LIST_START = re.compile(r"(?:^|\n)\s*1[.)]\s+", re.MULTILINE)


def _split_name_and_reasoning(raw_item):
    """Split a single parsed item into (name, reasoning_text).

    Handles shapes like:
        "**Basal Cell Carcinoma (BCC):** **Reasoning:** The image shows..."
        "Basal Cell Carcinoma: The lesion exhibits..."
        "Nevus - The lesion is small..."
        "Basal Cell Carcinoma\\n    * Reasoning: ..."
    """
    if not raw_item:
        return "", ""

    name_part, _, rest = raw_item.partition("\n")
    name_part = name_part.strip()
    rest = rest.strip()

    m = _NAME_REASON_SPLIT.search(name_part)
    if m:
        inline_reasoning = name_part[m.end():].strip()
        name_part = name_part[: m.start()].strip()
        if inline_reasoning:
            rest = (inline_reasoning + ("\n" + rest if rest else "")).strip()

    name = _extract_name(name_part)
    name = name.rstrip(":,;. ").strip()

    rest_lines = []
    for ln in rest.splitlines():
        ln = _LEADING_BULLET.sub("", ln).strip()
        ln = _REASONING_LABEL.sub("", ln).strip()
        if ln:
            rest_lines.append(ln)
    reasoning = " ".join(rest_lines).strip()
    reasoning = _REASONING_LABEL.sub("", reasoning).strip()
    return name, reasoning


def parse_reason_response(text, max_items=3):
    """Parse a reason-format VLM response into structured diagnoses.

    Returns a list of dicts:
        [{"name": str, "reasoning_sentences": list[str]}, ...]
    """
    if not text:
        return []

    # Reason-format responses usually begin with "1. ..." directly. When we see
    # that shape early in the text, parse the whole thing as a numbered list
    # instead of using split_response (which has fuzzy mid-sentence patterns
    # that can mis-split reasoning paragraphs).
    list_match = _REASON_LIST_START.search(text)
    if list_match and list_match.start() < 400:
        diag_raw = text
    else:
        _, diag_raw = split_response(text)
        if not diag_raw:
            diag_raw = text

    diag_clean = clean_markdown(diag_raw)
    items = parse_diagnoses(diag_clean)

    parsed = []
    for raw_item in items[:max_items]:
        name, reasoning = _split_name_and_reasoning(raw_item)
        if not name:
            continue
        sentences = split_sentences(reasoning) if reasoning else []
        parsed.append({"name": name, "reasoning_sentences": sentences})
    return parsed
