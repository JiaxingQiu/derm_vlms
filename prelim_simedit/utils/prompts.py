"""Prompt templates shared across the three phases.

Supports two modes via DIFFERENTIAL config:
  - "top_1": robot gives single top-1 diagnosis, judge gives its own top-1
  - "top_3": robot gives top-3 differential, judge gives its own top-3
"""

SYSTEM_JUDGE = (
    "You are an expert board-certified dermatologist acting as a reviewer. "
    "You assess another model's diagnosis for a skin lesion image and provide "
    "your own expert diagnosis."
)

# =============================================================================
# Phase 1: preedit
# =============================================================================

PREEDIT_TOP1 = (
    "Look at the lesion in this image and give your single most likely "
    "diagnosis (top-1 only). Respond with exactly one line:\n"
    "Diagnosis: <diagnosis name>"
)

PREEDIT_TOP3 = (
    "You are an expert dermatologist. Please give the top 3 diagnoses in "
    "your differential, and provide reasoning for each diagnosis, in format:\n"
    "1. [Diagnosis]: [Reasoning]\n"
    "2. [Diagnosis]: [Reasoning]\n"
    "3. [Diagnosis]: [Reasoning]"
)


def preedit_prompt(differential="top_1"):
    return PREEDIT_TOP1 if differential == "top_1" else PREEDIT_TOP3


# =============================================================================
# Phase 2: judge
# =============================================================================

JUDGE_TOP1 = (
    "A diagnostic model examined the lesion in this image and proposed the "
    "following diagnosis:\n\n"
    "    \"{dx}\"\n\n"
    "As an expert dermatologist, review this against what you see in the "
    "image. Provide your own diagnosis and briefly justify from visual "
    "findings.\n\n"
    "Respond ONLY with a JSON object, no extra text:\n"
    "{{\"diagnosis\": \"<your diagnosis>\", "
    "\"reasoning\": \"<one or two sentences>\"}}"
)

JUDGE_TOP3 = (
    "A diagnostic model examined the lesion in this image and proposed this "
    "differential diagnosis:\n\n"
    "{dx}\n\n"
    "As an expert dermatologist, review this differential against what you "
    "see in the image. Provide your own corrected top-3 differential in the "
    "same format: 1. [Diagnosis]: [Reasoning]\n\n"
    "Respond ONLY with a JSON object, no extra text:\n"
    "{{\"corrected_differential\": \"1. [Diagnosis]: [Reasoning]\\n"
    "2. [Diagnosis]: [Reasoning]\\n3. [Diagnosis]: [Reasoning]\"}}"
)


def judge_prompt(dx, differential="top_1"):
    if differential == "top_1":
        return JUDGE_TOP1.format(dx=dx)
    return JUDGE_TOP3.format(dx=dx)


# =============================================================================
# Phase 3: postedit — always show judge's diagnosis as hint
# =============================================================================

POSTEDIT_HINT_TOP1 = (
    "\n\nNote: A reviewing expert dermatologist suggests the diagnosis is "
    "\"{judge_dx}\" because: {reasoning}"
)

POSTEDIT_HINT_TOP3 = (
    "\n\nNote: A reviewing expert dermatologist suggests the following "
    "corrected differential:\n"
    "{corrected_differential}"
)


def postedit_prompt(differential="top_1", **kwargs):
    """Build postedit prompt: same question as preedit + judge's diagnosis as hint.

    For top_1: kwargs = judge_dx, reasoning
    For top_3: kwargs = corrected_differential
    """
    if differential == "top_1":
        base = PREEDIT_TOP1
        hint = POSTEDIT_HINT_TOP1.format(
            judge_dx=kwargs.get("judge_dx", ""),
            reasoning=kwargs.get("reasoning", ""),
        )
    else:
        base = PREEDIT_TOP3
        hint = POSTEDIT_HINT_TOP3.format(
            corrected_differential=kwargs.get("corrected_differential", ""),
        )

    return base + hint
