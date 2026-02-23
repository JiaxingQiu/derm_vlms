from typing import Tuple, Optional
import math
import re

def _clean_text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, str):
        v = value.strip()
        if v == "" or v.lower() in {"na", "n/a"}:
            return None
        return v
    return str(value)

def _fmt_age(age) -> Optional[str]:
    if age is None:
        return None
    try:
        f = float(age)
        return f"{int(f)}" if f.is_integer() else f"{f:.0f}"
    except Exception:
        return _clean_text(age)

def _fmt_size(length, width) -> Optional[str]:
    l = None if length is None else float(length)
    w = None if width is None else float(width)
    if l is not None and w is not None:
        return f"{l:.1f} × {w:.1f} mm"
    if l is not None:
        return f"{l:.1f} mm (length)"
    if w is not None:
        return f"{w:.1f} mm (width)"
    return None

# helpers (place once, outside the function)
def _as_lower_str(value) -> Optional[str]:
    t = _clean_text(value)
    return t.lower() if t else None

def _fmt_melanoma_history(value) -> Optional[str]:
    v = _as_lower_str(value)
    if v in {"yes", "y", "true", "1"}:
        return "The patient has a history of melanoma"
    if v in {"no", "n", "false", "0"}:
        return "The patient has no history of melanoma"
    return None  # omit unknown/blank

def _fmt_ethnicity(value) -> Optional[str]:
    v = _as_lower_str(value)
    if v in {"yes", "y", "true", "1", "hispanic", "latino", "hispanic/latino"}:
        return "Hispanic/Latino"
    return None  # suppress 'no' and unknown

def _humanize_impression(text: Optional[str]) -> Optional[str]:
    t = _clean_text(text)
    if not t:
        return None
    # Drop leading numeric codes like "7-" and replace hyphens with spaces where appropriate
    t = re.sub(r"^\d+\s*-\s*", "", t)
    t = t.replace("_", " ")
    # Keep known abbreviations intact; expand simple ones if desired
    t = t.replace("bcc", "BCC").replace("scc", "SCC")
    return t

def _dedupe_periods(text: Optional[str]) -> Optional[str]:
    """Collapse any run of 2+ periods to a single period."""
    if not text:
        return text
    return re.sub(r"\.{2,}", ".", text)

def row_to_natural_text(row) -> Tuple[str, str, str, str]:
    """
    Outcome: This patient’s diagnosis is {y16} ({y3}). {y16_description} {path_report_sentence}
    Predictors: The patient has {x_skintype}. The lesion is on the {x_location}.
    Demographics: The patient is {age_sentence}{gender_sentence}{fitz_sentence}{melanoma_history_sentence}{race_eth_sentence}.
    Lesion: The lesion is on the {lesion_location}, imaged at {lesion_distance}. It measures {len_width_mm_sentence}. Clinical impressions include {impressions_sentence}.
    """
    # Outcome
    y16 = _clean_text(row.get("y16"))
    y3 = _clean_text(row.get("y3"))
    # y3 = y3.replace("other", "unknown")
    # y16_desc = _clean_text(row.get("y16_description"))
    path = _clean_text(row.get("notes_pathreport"))

    text_y3  = f"The lesion is {y3}." if y3 else ""
    text_y16 = f"The patient is diagnosed with {_as_lower_str(y16)}" if y16 else ""
    text_path = f"On pathology, {path.rstrip('.')}." if path else ""
    parts = [text_y3, text_y16, text_path]
    text_outcome = " ".join([p for p in parts if p]).strip()

    # PREDICTORS
    x_skintype = _clean_text(row.get("x_skintype"))
    x_skincolor = _clean_text(row.get("x_skincolor"))
    x_skintone = _clean_text(row.get("x_skintone"))
    x_location = _clean_text(row.get("x_location"))
    text_x_skintype = f"The patient has {x_skintype}." if x_skintype else ""
    text_x_skincolor = f"The patient has {x_skincolor} skin color." if x_skincolor and x_skincolor != "unknown" else ""
    text_x_skintone = f"The patient has {x_skintone} skin tone." if x_skintone and x_skintone != "unknown" else ""
    text_x_location = f"The lesion is on the {x_location}." if x_location else ""
    
    # DEMOGRAPHICS
    age = _fmt_age(row.get("demo_age"))
    gender = _clean_text(row.get("demo_gender"))
    fitz = _clean_text(row.get("demo_fitzpatrick_skintype"))
    race = _clean_text(row.get("demo_race"))
    eth_phrase = _fmt_ethnicity(row.get("demo_ethnicity"))  # e.g., "Hispanic/Latino" or None
    hist_phrase = _fmt_melanoma_history(row.get("demo_melanoma_history"))  # e.g., "with a history of melanoma"

    parts = []
    intro_bits = []

    if age:
        intro_bits.append(f"{age}-year-old")
    if eth_phrase:
        intro_bits.append(eth_phrase)
    if race:
        intro_bits.append(race)
    if gender:
        intro_bits.append(gender)

    if intro_bits:
        parts.append("The patient is " + " ".join(intro_bits))
    if fitz:
        parts.append(f"with Fitzpatrick type {fitz}.")
    if hist_phrase:
        parts.append(hist_phrase)

    demographics_text = ""
    if parts:
        sentence = " ".join(parts).strip()
        if not sentence.endswith("."):
            sentence += "."
        demographics_text = sentence

    # Lesion
    loc = _clean_text(row.get("lesion_location"))
    dist = _clean_text(row.get("lesion_distance"))
    size_txt = _fmt_size(row.get("lesion_length_mm"), row.get("lesion_width_mm"))

    imps = [
        _humanize_impression(row.get("notes_clinical_impression_1")),
        _humanize_impression(row.get("notes_clinical_impression_2")),
        _humanize_impression(row.get("notes_clinical_impression_3")),
    ]
    imps = [i for i in imps if i]
    imps_txt = None
    if imps:
        # Oxford comma style for 3+ items
        if len(imps) == 1:
            imps_txt = imps[0]
        elif len(imps) == 2:
            imps_txt = " and ".join(imps)
        else:
            imps_txt = ", ".join(imps[:-1]) + f", and {imps[-1]}"

    lesion_bits = []
    if loc and dist:
        lesion_bits.append(f"The lesion is on the {loc}, imaged at {dist}.")
    elif loc:
        lesion_bits.append(f"The lesion is on the {loc}.")
    elif dist:
        lesion_bits.append(f"The lesion was imaged at {dist}.")
    if size_txt:
        lesion_bits.append(f"It measures {size_txt}.")
    if imps_txt:
        lesion_bits.append(f"Clinical impressions include {imps_txt}.")
    lesion_text = " ".join(lesion_bits).strip()

    # Full paragraph
    full = " ".join([t for t in [text_x_skintype, text_x_skincolor, text_x_skintone, text_x_location, text_outcome] if t]).strip()

    # Safeguard: remove accidental double periods in all returned texts
    full = _dedupe_periods(full)
    text_outcome = _dedupe_periods(text_outcome)
    text_x_skincolor = _dedupe_periods(text_x_skincolor)
    text_x_skintone = _dedupe_periods(text_x_skintone)
    demographics_text = _dedupe_periods(demographics_text)
    lesion_text = _dedupe_periods(lesion_text)

    return full, text_outcome, text_y3, text_y16, text_x_skintype, text_x_skincolor, text_x_skintone, text_x_location, demographics_text, lesion_text

# Usage:
# full, text_outcome, text_y3, text_y16, text_x_skintype, text_x_skincolor, text_x_skintone, text_x_location, demographics_text, lesion_text = row_to_natural_text(df.iloc[0])