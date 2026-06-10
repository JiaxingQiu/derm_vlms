"""RCT case-assignment logic — static slot-based design.

A fixed pool of 500 "slots" is pre-computed once (by ``parsedata``) and
written to ``data/assignment_slots.json``.  Each slot contains a
deterministic list of case_ids.  When a real user registers, they claim
the next unclaimed slot and its case list is copied into Assignment rows.

The slot file is **idempotent**: re-running generation with the same
eligible lesions and seed always produces a byte-identical file.

See ``ASSIGNMENT.md`` for the full design rationale.
"""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
ANCHOR_LESION = "1025"
MAX_SLOTS = 500

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
    """Pick *n* lesions for one user using bucket-based priority sampling.

    The anchor lesion (ANCHOR_LESION) is always included first as a
    warmup/calibration case.  The remaining n-1 slots are filled from
    the pool excluding the anchor.
    """
    has_anchor = ANCHOR_LESION in eligible_lesions
    pool = [lid for lid in eligible_lesions if lid != ANCHOR_LESION]
    n_random = (n - 1) if has_anchor else n

    need_more = []
    fresh = []
    saturated = []

    for lid in pool:
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
    selected.extend(need_more[:min(len(need_more), need_more_cap, n_random)])
    remaining = n_random - len(selected)
    if remaining > 0:
        selected.extend(fresh[:remaining])
    remaining = n_random - len(selected)
    if remaining > 0:
        selected.extend(saturated[:remaining])

    selected_set = set(selected)
    ordered = [lid for lid in eligible_lesions if lid in selected_set and lid != ANCHOR_LESION]

    if has_anchor:
        return [ANCHOR_LESION] + ordered
    return ordered


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


# =========================================================================
# Static slot-based assignment
# =========================================================================

SLOTS_FILENAME = "assignment_slots.json"


def _slots_path():
    return Path(settings.BASE_DIR) / "data" / SLOTS_FILENAME


def generate_all_slots(eligible_lesions, n_slots=MAX_SLOTS,
                       seed=DEFAULT_SEED,
                       enabled_factors=DEFAULT_ENABLED_FACTORS,
                       out_path=None):
    """Pre-compute *n_slots* deterministic case lists and write to JSON.

    Each slot uses a synthetic user id ``slot_000`` … ``slot_499`` so
    the output is reproducible regardless of real user names.  Slot 0
    is reserved for test accounts.

    Returns the path to the written file.
    """
    if out_path is None:
        out_path = _slots_path()
    out_path = Path(out_path)

    lesion_counts = Counter()
    slots = []

    for i in range(n_slots):
        user_id = f"slot_{i:03d}"
        cfg = ROLE_CONFIG["Dermatologist"]

        case_list, _ = build_case_list_for_user(
            user_id, eligible_lesions, lesion_counts,
            seed, cfg["lesions_per_user"], enabled_factors,
            cfg["min_annotators"], cfg["max_annotators"],
            cfg["need_more_cap"],
        )

        for _order, case_id in case_list:
            lid = case_id.rsplit("_", 1)[0]
            lesion_counts[lid] += 1

        slots.append({
            "slot": i,
            "case_ids": [cid for _order, cid in case_list],
        })

    payload = {
        "seed": seed,
        "n_slots": n_slots,
        "eligible_lesion_count": len(eligible_lesions),
        "cases_per_slot": cfg["lesions_per_user"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slots": slots,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path


def load_assignment_slots(path=None):
    """Load the pre-computed slot file.  Returns the full dict."""
    if path is None:
        path = _slots_path()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def next_available_slot(role="Dermatologist"):
    """Return the lowest slot number not yet claimed by any user.

    Reads the ``assignment_slot`` field from the user table.
    """
    from .models import Dermatologist, PCPUser
    UserModel = PCPUser if role == "PCP" else Dermatologist
    used = set(
        UserModel.objects
        .exclude(assignment_slot__isnull=True)
        .values_list("assignment_slot", flat=True)
    )
    slots_data = load_assignment_slots()
    for entry in slots_data["slots"]:
        if entry["slot"] not in used:
            return entry["slot"]
    raise RuntimeError(
        f"All {slots_data['n_slots']} assignment slots are claimed for {role}. "
        "Generate more slots or increase MAX_SLOTS."
    )


def assign_from_slot(evaluator, slot_number, role="Dermatologist"):
    """Write Assignment rows for *evaluator* using a pre-computed slot.

    This is the **only** way real users get assignments at registration.
    It reads from the static JSON file and never recomputes anything.
    """
    AssignmentModel = _get_assignment_model(role)
    slots_data = load_assignment_slots()

    slot_entry = slots_data["slots"][slot_number]
    assert slot_entry["slot"] == slot_number

    AssignmentModel.objects.filter(evaluator=evaluator).delete()
    AssignmentModel.objects.bulk_create([
        AssignmentModel(evaluator=evaluator, case_id=case_id, order=order)
        for order, case_id in enumerate(slot_entry["case_ids"])
    ])

    evaluator.assignment_slot = slot_number
    evaluator.save(update_fields=["assignment_slot"])
