"""RCT case-assignment logic, shared by registration and the management command.

The core function :func:`assign_cases_for_user` implements a latin-square
rotation across eligible lesions so that each evaluator sees a balanced
mix of image modes.  It creates :class:`Assignment` rows in the database.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from django.conf import settings

FACTORS = {
    "image_mode": {
        "levels": ["photo", "dscope", "combined"],
        "default": "combined",
    },
}

DEFAULT_SEED = 42
DEFAULT_MAX_LESIONS = 5
DEFAULT_ENABLED_FACTORS = ("image_mode",)


def _stable_hash(seed, *parts):
    blob = "|".join(str(p) for p in (seed, *parts))
    return int(hashlib.sha256(blob.encode()).hexdigest(), 16)


def _user_factor_group(seed, user_id, factor_name, n_levels):
    return _stable_hash(seed, "group", factor_name, user_id) % n_levels


def _assign_level(factor_group, lesion_index, n_levels):
    return (factor_group + lesion_index) % n_levels


def get_eligible_lesions(annotations_data=None):
    """Return sorted list of lesion IDs that have all 3 image modes."""
    if annotations_data is None:
        json_path = Path(settings.BASE_DIR) / "data" / "annotations_data.json"
        with open(json_path, "r", encoding="utf-8") as f:
            annotations_data = json.load(f)

    lesion_modes = defaultdict(set)
    for case_id in annotations_data:
        parts = case_id.rsplit("_", 1)
        if len(parts) == 2:
            lesion_id, mode = parts
            lesion_modes[lesion_id].add(mode)

    required_modes = set(FACTORS["image_mode"]["levels"])
    return sorted(lid for lid, modes in lesion_modes.items()
                  if required_modes.issubset(modes))


def build_case_list_for_user(user_id, eligible_lesions, seed=DEFAULT_SEED,
                             max_lesions=DEFAULT_MAX_LESIONS,
                             enabled_factors=DEFAULT_ENABLED_FACTORS):
    """Compute the ordered list of case_ids for one user (pure logic, no DB).

    Returns a list of ``(order, case_id)`` tuples.
    """
    enabled_factors = set(enabled_factors)
    lesions = eligible_lesions
    if max_lesions and max_lesions < len(lesions):
        lesions = lesions[:max_lesions]

    factor_groups = {}
    for factor_name, spec in FACTORS.items():
        n = len(spec["levels"])
        factor_groups[factor_name] = _user_factor_group(seed, user_id, factor_name, n)

    result = []
    for li, lesion_id in enumerate(lesions):
        conditions = {}
        for factor_name, spec in FACTORS.items():
            if factor_name in enabled_factors:
                idx = _assign_level(
                    factor_groups[factor_name], li, len(spec["levels"])
                )
                conditions[factor_name] = spec["levels"][idx]
            else:
                conditions[factor_name] = spec["default"]

        case_id = f"{lesion_id}_{conditions['image_mode']}"
        result.append((li, case_id))

    return result, factor_groups


def assign_cases_for_user(evaluator, eligible_lesions=None, seed=DEFAULT_SEED,
                          max_lesions=DEFAULT_MAX_LESIONS,
                          enabled_factors=DEFAULT_ENABLED_FACTORS):
    """Create :class:`Assignment` rows for one evaluator.

    If the evaluator already has assignments they are replaced.
    Returns the list of created Assignment objects.
    """
    from .models import Assignment

    if eligible_lesions is None:
        eligible_lesions = get_eligible_lesions()

    case_list, _ = build_case_list_for_user(
        evaluator.login_id, eligible_lesions, seed, max_lesions, enabled_factors,
    )

    Assignment.objects.filter(evaluator=evaluator).delete()
    assignments = Assignment.objects.bulk_create([
        Assignment(evaluator=evaluator, case_id=case_id, order=order)
        for order, case_id in case_list
    ])
    return assignments
