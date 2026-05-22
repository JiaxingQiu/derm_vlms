"""RCT case-assignment logic, shared by registration and the management command.

Each user is assigned a configurable number of lesions per role.  Lesions are
chosen so that (a) coverage across the full pool is maximised, and (b) any
lesion seen by more than one user ends up with the target annotator range.

Assignment is **sequential**: each new user's selection depends on the counts
accumulated by all previously registered users.  Reproducibility is
guaranteed by always processing users in ``registered_at`` order and using
a deterministic per-user seed.
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from django.conf import settings

FACTORS = {
    "image_mode": {
        "levels": ["photo", "dscope", "combined"],
        "default": "combined",
    },
}

DEFAULT_SEED = 42
DEFAULT_ENABLED_FACTORS = ()

# Legacy module-level constants kept for backward compatibility.
LESIONS_PER_USER = 101
MIN_ANNOTATORS = 3
MAX_ANNOTATORS = 5
NEED_MORE_CAP = 50

ROLE_CONFIG = {
    "Dermatologist": {
        "lesions_per_user": 101,
        "min_annotators": 3,
        "max_annotators": 5,
        "need_more_cap": 50,
    },
    "PCP": {
        "lesions_per_user": 101,
        "min_annotators": 3,
        "max_annotators": 5,
        "need_more_cap": 50,
    },
}


def _get_role_config(role):
    return ROLE_CONFIG.get(role, ROLE_CONFIG["Dermatologist"])


def _get_assignment_model(role):
    from .models import Assignment, PCPAssignment
    if role == "PCP":
        return PCPAssignment
    return Assignment


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


def get_lesion_counts_from_db(exclude_evaluator=None, role="Dermatologist"):
    """Query existing Assignment rows and return {lesion_id: annotator_count}.

    The returned Counter covers only lesion-level IDs (mode suffix stripped).
    Counts are scoped to the given role's assignment table.
    """
    AssignmentModel = _get_assignment_model(role)

    qs = AssignmentModel.objects.all()
    if exclude_evaluator is not None:
        qs = qs.exclude(evaluator=exclude_evaluator)

    counts = Counter()
    seen = set()
    for evaluator_id, case_id in qs.values_list("evaluator_id", "case_id"):
        parts = case_id.rsplit("_", 1)
        lid = parts[0] if len(parts) == 2 else case_id
        key = (evaluator_id, lid)
        if key not in seen:
            seen.add(key)
            counts[lid] += 1
    return counts


def _select_lesions_for_user(user_id, eligible_lesions, lesion_counts,
                             seed=DEFAULT_SEED, n=LESIONS_PER_USER,
                             min_annotators=MIN_ANNOTATORS,
                             max_annotators=MAX_ANNOTATORS,
                             need_more_cap=NEED_MORE_CAP):
    """Pick *n* lesions for one user using bucket-based priority sampling."""
    need_more = []
    fresh = []
    saturated = []

    for lid in eligible_lesions:
        c = lesion_counts.get(lid, 0)
        if c == 0:
            fresh.append(lid)
        elif c < min_annotators:
            need_more.append(lid)
        elif c < max_annotators:
            saturated.append(lid)

    def _seeded_sort(bucket, tag):
        return sorted(bucket, key=lambda lid: _stable_hash(seed, tag, user_id, lid))

    need_more = _seeded_sort(need_more, "need_more")
    fresh = _seeded_sort(fresh, "fresh")
    saturated = _seeded_sort(saturated, "saturated")

    selected = []
    selected.extend(need_more[:min(len(need_more), need_more_cap, n)])
    remaining = n - len(selected)
    if remaining > 0:
        selected.extend(fresh[:remaining])
    remaining = n - len(selected)
    if remaining > 0:
        selected.extend(saturated[:remaining])

    selected_set = set(selected)
    return [lid for lid in eligible_lesions if lid in selected_set]


def build_case_list_for_user(user_id, eligible_lesions, lesion_counts,
                             seed=DEFAULT_SEED,
                             n=LESIONS_PER_USER,
                             enabled_factors=DEFAULT_ENABLED_FACTORS,
                             min_annotators=MIN_ANNOTATORS,
                             max_annotators=MAX_ANNOTATORS,
                             need_more_cap=NEED_MORE_CAP):
    """Compute the ordered list of case_ids for one user (pure logic, no DB).

    Returns ``([(order, case_id), ...], factor_groups)``.
    """
    enabled_factors = set(enabled_factors)
    lesions = _select_lesions_for_user(
        user_id, eligible_lesions, lesion_counts, seed, n,
        min_annotators, max_annotators, need_more_cap,
    )

    factor_groups = {}
    for factor_name, spec in FACTORS.items():
        n_levels = len(spec["levels"])
        factor_groups[factor_name] = _user_factor_group(seed, user_id, factor_name, n_levels)

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
                          n=None, enabled_factors=DEFAULT_ENABLED_FACTORS,
                          role="Dermatologist"):
    """Create Assignment/PCPAssignment rows for one evaluator.

    Reads current lesion counts from the DB (excluding this evaluator's own
    prior assignments, which are deleted first).  Returns the list of created
    Assignment objects.
    """
    cfg = _get_role_config(role)
    AssignmentModel = _get_assignment_model(role)

    if n is None:
        n = cfg["lesions_per_user"]

    if eligible_lesions is None:
        eligible_lesions = get_eligible_lesions()

    AssignmentModel.objects.filter(evaluator=evaluator).delete()

    lesion_counts = get_lesion_counts_from_db(
        exclude_evaluator=evaluator, role=role,
    )

    case_list, _ = build_case_list_for_user(
        evaluator.login_id, eligible_lesions, lesion_counts,
        seed, n, enabled_factors,
        cfg["min_annotators"], cfg["max_annotators"], cfg["need_more_cap"],
    )

    assignments = AssignmentModel.objects.bulk_create([
        AssignmentModel(evaluator=evaluator, case_id=case_id, order=order)
        for order, case_id in case_list
    ])
    return assignments


def regenerate_all_assignments(evaluators, eligible_lesions, seed=DEFAULT_SEED,
                               n=None, enabled_factors=DEFAULT_ENABLED_FACTORS,
                               dry_run=False, role="Dermatologist"):
    """Deterministically (re)generate assignments for a list of evaluators.

    Processes evaluators in the given order, simulating lesion counts
    in-memory so the result is identical regardless of current DB state.

    Yields ``(evaluator, case_list)`` tuples.  When *dry_run* is False,
    also writes to the DB.
    """
    cfg = _get_role_config(role)
    AssignmentModel = _get_assignment_model(role)

    if n is None:
        n = cfg["lesions_per_user"]

    lesion_counts = Counter()

    for evaluator in evaluators:
        case_list, _ = build_case_list_for_user(
            evaluator.login_id, eligible_lesions, lesion_counts,
            seed, n, enabled_factors,
            cfg["min_annotators"], cfg["max_annotators"], cfg["need_more_cap"],
        )

        for _order, case_id in case_list:
            lid = case_id.rsplit("_", 1)[0]
            lesion_counts[lid] += 1

        if not dry_run:
            AssignmentModel.objects.filter(evaluator=evaluator).delete()
            AssignmentModel.objects.bulk_create([
                AssignmentModel(evaluator=evaluator, case_id=case_id, order=order)
                for order, case_id in case_list
            ])

        yield evaluator, case_list
