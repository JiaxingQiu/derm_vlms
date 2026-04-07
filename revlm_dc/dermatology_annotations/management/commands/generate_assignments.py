"""Generate reproducible RCT assignments for the annotation interface.

Each user is deterministically assigned one image mode per lesion using
a Latin-square rotation.  Multiple RCT factors are supported; each
factor is rotated independently so that factors remain orthogonal.

Usage:
    python manage.py generate_assignments assignments.yaml
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
from django.core.management.base import BaseCommand, CommandError

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
            "config",
            nargs="?",
            help="Path to a YAML config file containing users and command options.",
        )
        parser.add_argument(
            "--users",
            nargs="+",
            help="Login IDs of users to include in the study.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            help="Global random seed for reproducibility (default: 42).",
        )
        parser.add_argument(
            "--max-lesions",
            type=int,
            help="Max lesions per user (default: all eligible lesions).",
        )
        parser.add_argument(
            "--enable-factors",
            nargs="+",
            choices=list(FACTORS.keys()),
            help="Which RCT factors to randomize (default: image_mode).",
        )

    def _load_yaml_config(self, config_path):
        try:
            import yaml
        except ImportError as exc:
            raise CommandError(
                "YAML config support requires PyYAML. Install it or use CLI flags."
            ) from exc

        path = Path(config_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise CommandError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise CommandError("Config file must contain a YAML mapping at the top level.")

        allowed_keys = {"users", "seed", "max_lesions", "enable_factors"}
        alias_keys = {
            "max-lesions": "max_lesions",
            "enable-factors": "enable_factors",
        }
        normalized = {}
        for key, value in data.items():
            normalized_key = alias_keys.get(key, key)
            if normalized_key not in allowed_keys:
                raise CommandError(
                    f"Unsupported config key '{key}'. Allowed keys: {sorted(allowed_keys)}"
                )
            normalized[normalized_key] = value

        return path, normalized

    def _resolve_options(self, options):
        config_data = {}
        config_path = options.get("config")
        if config_path:
            resolved_path, config_data = self._load_yaml_config(config_path)
            self.stdout.write(f"Loading assignment config from {resolved_path}")

        resolved = {
            "users": options.get("users") if options.get("users") is not None else config_data.get("users"),
            "seed": options.get("seed") if options.get("seed") is not None else config_data.get("seed", 42),
            "max_lesions": (
                options.get("max_lesions")
                if options.get("max_lesions") is not None
                else config_data.get("max_lesions")
            ),
            "enable_factors": (
                options.get("enable_factors")
                if options.get("enable_factors") is not None
                else config_data.get("enable_factors", ["image_mode"])
            ),
        }

        if not resolved["users"]:
            raise CommandError(
                "No users provided. Pass --users or define 'users' in the YAML config."
            )
        if not isinstance(resolved["users"], list) or not all(
            isinstance(user, str) and user for user in resolved["users"]
        ):
            raise CommandError("'users' must be a non-empty list of user IDs.")
        if not isinstance(resolved["seed"], int):
            raise CommandError("'seed' must be an integer.")
        if resolved["max_lesions"] is not None and (
            not isinstance(resolved["max_lesions"], int) or resolved["max_lesions"] <= 0
        ):
            raise CommandError("'max_lesions' must be a positive integer or null.")
        if not isinstance(resolved["enable_factors"], list) or not resolved["enable_factors"]:
            raise CommandError("'enable_factors' must be a non-empty list.")

        invalid_factors = sorted(set(resolved["enable_factors"]) - set(FACTORS.keys()))
        if invalid_factors:
            raise CommandError(
                f"Unknown factors in enable_factors: {invalid_factors}. "
                f"Allowed: {sorted(FACTORS.keys())}"
            )

        return resolved

    def handle(self, *args, **options):
        resolved_options = self._resolve_options(options)
        seed = resolved_options["seed"]
        user_ids = resolved_options["users"]
        max_lesions = resolved_options["max_lesions"]
        enabled_factors = set(resolved_options["enable_factors"])

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
