"""Generate reproducible RCT assignments for the annotation interface.

Each user is deterministically assigned one image mode per lesion using
a Latin-square rotation.  Multiple RCT factors are supported; each
factor is rotated independently so that factors remain orthogonal.

Usage:
    python manage.py generate_assignments --users user_a user_b user_c
    python manage.py generate_assignments --users user_a user_b --seed 42 --max-lesions 50
    python manage.py generate_assignments --users user_a user_b \
        --enable-factors image_mode interface_type

The command also exports a flat CSV (data/assignments.csv) for
downstream statistical analysis.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

FACTORS = {
    "image_mode": {
        "levels": ["photo", "dscope", "combined"],
        "default": "combined",
    },
    "interface_type": {
        "levels": ["conditional", "unconditional"],
        "default": "conditional",
    },
}


def _stable_hash(seed, *parts):
    """Deterministic integer hash from seed + arbitrary string parts."""
    blob = "|".join(str(p) for p in (seed, *parts))
    return int(hashlib.sha256(blob.encode()).hexdigest(), 16)


def _user_factor_group(seed, user_id, factor_name, n_levels):
    """Assign a user to a factor group in [0, n_levels)."""
    return _stable_hash(seed, "group", factor_name, user_id) % n_levels


def _assign_level(factor_group, lesion_index, n_levels):
    """Latin-square rotation: level for this lesion given the user's group."""
    return (factor_group + lesion_index) % n_levels


class Command(BaseCommand):
    help = "Generate reproducible per-user RCT assignments and export CSV."

    def add_arguments(self, parser):
        parser.add_argument(
            "--users",
            nargs="+",
            required=True,
            help="Login IDs of users to include in the study.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Global random seed for reproducibility (default: 42).",
        )
        parser.add_argument(
            "--max-lesions",
            type=int,
            default=None,
            help="Max lesions per user (default: all eligible lesions).",
        )
        parser.add_argument(
            "--enable-factors",
            nargs="+",
            default=["image_mode"],
            choices=list(FACTORS.keys()),
            help="Which RCT factors to randomize (default: image_mode).",
        )

    def handle(self, *args, **options):
        seed = options["seed"]
        user_ids = options["users"]
        max_lesions = options["max_lesions"]
        enabled_factors = set(options["enable_factors"])

        data_dir = Path(settings.BASE_DIR) / "data"
        annotations_path = data_dir / "annotations_data.json"

        if not annotations_path.exists():
            self.stdout.write(self.style.ERROR(
                f"{annotations_path} not found. Run parsedata first."
            ))
            return

        with open(annotations_path, "r", encoding="utf-8") as f:
            annotations_data = json.load(f)

        lesion_modes = defaultdict(set)
        for case_id in annotations_data:
            parts = case_id.rsplit("_", 1)
            if len(parts) == 2:
                lesion_id, mode = parts
                lesion_modes[lesion_id].add(mode)

        required_modes = set(FACTORS["image_mode"]["levels"])
        eligible_lesions = sorted(
            lid for lid, modes in lesion_modes.items()
            if required_modes.issubset(modes)
        )
        self.stdout.write(
            f"Eligible lesions (have all 3 modes): {len(eligible_lesions)}"
        )

        if not eligible_lesions:
            self.stdout.write(self.style.ERROR("No eligible lesions. Aborting."))
            return

        if max_lesions and max_lesions < len(eligible_lesions):
            eligible_lesions = eligible_lesions[:max_lesions]
            self.stdout.write(f"Capped to --max-lesions={max_lesions}")

        users_out = {}
        csv_rows = []

        for user_id in user_ids:
            factor_groups = {}
            for factor_name, spec in FACTORS.items():
                n = len(spec["levels"])
                factor_groups[factor_name] = _user_factor_group(
                    seed, user_id, factor_name, n
                )

            assignments = []
            for li, lesion_id in enumerate(eligible_lesions):
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

                assignments.append({
                    "lesion_id": lesion_id,
                    "case_id": case_id,
                    "conditions": conditions,
                })

                csv_rows.append({
                    "user_id": user_id,
                    **{f"factor_group_{k}": v for k, v in factor_groups.items()},
                    "lesion_id": lesion_id,
                    "case_id": case_id,
                    **conditions,
                })

            users_out[user_id] = {
                "factor_groups": factor_groups,
                "assignments": assignments,
            }

            mode_counts = defaultdict(int)
            for a in assignments:
                mode_counts[a["conditions"]["image_mode"]] += 1
            self.stdout.write(
                f"  {user_id}: {len(assignments)} lesions, "
                f"groups={factor_groups}, mode dist={dict(mode_counts)}"
            )

        output = {
            "seed": seed,
            "factors": {
                name: {
                    "levels": spec["levels"],
                    "enabled": name in enabled_factors,
                }
                for name, spec in FACTORS.items()
            },
            "max_lesions_per_user": max_lesions,
            "eligible_lesion_count": len(eligible_lesions),
            "users": users_out,
        }

        users_path = data_dir / "users.json"
        with open(users_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        self.stdout.write(self.style.SUCCESS(f"\nWrote {users_path}"))

        csv_path = data_dir / "assignments.csv"
        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            self.stdout.write(self.style.SUCCESS(
                f"Wrote {csv_path} ({len(csv_rows)} rows)"
            ))

        self.stdout.write(self.style.SUCCESS(
            "\nDone. Run: python manage.py runserver"
        ))
