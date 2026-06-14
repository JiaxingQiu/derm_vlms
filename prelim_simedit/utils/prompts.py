"""Prompt templates shared across the three phases.

Kept deliberately simple: the robot returns a single top-1 diagnosis, the
judge returns strict JSON. The judge never sees the ground-truth label.
"""

SYSTEM_ROBOT = (
    "You are a dermatology diagnosis assistant. Answer concisely using "
    "standard dermatological terminology."
)

SYSTEM_JUDGE = (
    "You are an expert board-certified dermatologist acting as a reviewer. "
    "You assess another model's diagnosis for a skin lesion image and decide "
    "whether it is correct."
)

# --- Phase 1: preedit -------------------------------------------------------

PREEDIT = (
    "Look at the lesion in this image and give your single most likely "
    "diagnosis (top-1 only). Respond with exactly one line:\n"
    "Diagnosis: <diagnosis name>"
)


def preedit_prompt():
    return PREEDIT


# --- Phase 2: judge ---------------------------------------------------------

JUDGE = (
    "A diagnostic model examined the lesion in this image and proposed the "
    "following diagnosis:\n\n"
    "    \"{dx}\"\n\n"
    "As an expert dermatologist, review this against what you see in the "
    "image. Decide whether the proposed diagnosis is correct. If it is not "
    "correct, state the correct diagnosis. Briefly justify your decision "
    "from the visual findings.\n\n"
    "Respond ONLY with a JSON object, no extra text:\n"
    "{{\"verdict\": \"correct\" | \"incorrect\", "
    "\"correct_diagnosis\": \"<diagnosis name>\", "
    "\"reasoning\": \"<one or two sentences>\"}}\n"
    "If the proposed diagnosis is correct, set correct_diagnosis to that same "
    "diagnosis."
)


def judge_prompt(dx):
    return JUDGE.format(dx=dx)


# --- Phase 3: postedit ------------------------------------------------------

POSTEDIT_HINT_INCORRECT = (
    "\n\nNote: A reviewing expert dermatologist has indicated that "
    "\"{preedit_dx}\" is not correct. The correct diagnosis is "
    "\"{correct_dx}\" because: {reasoning}"
)

POSTEDIT_HINT_CORRECT = (
    "\n\nNote: A reviewing expert dermatologist has confirmed that "
    "\"{preedit_dx}\" is correct."
)


def postedit_prompt(preedit_dx, verdict, correct_dx, reasoning):
    """Same question as preedit + judge feedback appended as extra context."""
    if str(verdict).strip().lower() == "correct":
        hint = POSTEDIT_HINT_CORRECT.format(preedit_dx=preedit_dx)
    else:
        hint = POSTEDIT_HINT_INCORRECT.format(
            preedit_dx=preedit_dx,
            correct_dx=correct_dx,
            reasoning=reasoning,
        )
    return PREEDIT + hint
