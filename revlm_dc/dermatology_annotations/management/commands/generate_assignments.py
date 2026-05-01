"""Generate reproducible RCT assignments for the annotation interface.

Each user is deterministically assigned one image mode per lesion using
a Latin-square rotation.  Assignments are written to the database
(:class:`Assignment` model) and optionally exported as a flat CSV.

Usage:
    python manage.py generate_assignments configs/test_config.yaml
    python manage.py generate_assignments --users user_a user_b user_c
    python manage.py generate_assignments --users user_a user_b --seed 42 --max-lesions 5
    python manage.py generate_assignments --users user_a user_b \\
        --enable-factors image_mode
"""

import csv
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from dermatology_annotations.assignments import (
    FACTORS,
    DEFAULT_SEED,
    DEFAULT_MAX_LESIONS,
    DEFAULT_ENABLED_FACTORS,
    assign_cases_for_user,
    build_case_list_for_user,
    get_eligible_lesions,
    _user_factor_group,
)
from dermatology_annotations.models import Dermatologist


class Command(BaseCommand):
    help = "Generate reproducible per-user RCT assignments (written to DB)."

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
            help="Max lesions per user (default: 5).",
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
            "seed": options.get("seed") if options.get("seed") is not None else config_data.get("seed", DEFAULT_SEED),
            "max_lesions": (
                options.get("max_lesions")
                if options.get("max_lesions") is not None
                else config_data.get("max_lesions", DEFAULT_MAX_LESIONS)
            ),
            "enable_factors": (
                options.get("enable_factors")
                if options.get("enable_factors") is not None
                else config_data.get("enable_factors", list(DEFAULT_ENABLED_FACTORS))
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
        enabled_factors = tuple(resolved_options["enable_factors"])

        eligible_lesions = get_eligible_lesions()
        self.stdout.write(
            f"Eligible lesions (have all 3 modes): {len(eligible_lesions)}"
        )
        if not eligible_lesions:
            self.stdout.write(self.style.ERROR("No eligible lesions. Aborting."))
            return

        csv_rows = []

        for user_id in user_ids:
            login_id = user_id.strip().lower()
            evaluator, created = Dermatologist.objects.get_or_create(
                login_id=login_id,
            )
            if created:
                self.stdout.write(f"  Created Dermatologist: {login_id}")

            assignments = assign_cases_for_user(
                evaluator, eligible_lesions, seed, max_lesions, enabled_factors,
            )

            factor_groups = {}
            for factor_name, spec in FACTORS.items():
                factor_groups[factor_name] = _user_factor_group(
                    seed, login_id, factor_name, len(spec["levels"]),
                )

            case_list, _ = build_case_list_for_user(
                login_id, eligible_lesions, seed, max_lesions, enabled_factors,
            )
            for order, case_id in case_list:
                lesion_id = case_id.rsplit("_", 1)[0]
                image_mode = case_id.rsplit("_", 1)[1] if "_" in case_id else ""
                csv_rows.append({
                    "user_id": login_id,
                    **{f"factor_group_{k}": v for k, v in factor_groups.items()},
                    "lesion_id": lesion_id,
                    "case_id": case_id,
                    "image_mode": image_mode,
                })

            self.stdout.write(
                f"  {login_id}: {len(assignments)} assignments, "
                f"groups={factor_groups}"
            )

        data_dir = Path(settings.BASE_DIR) / "data"
        data_dir.mkdir(exist_ok=True)
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
            f"\nDone. {len(user_ids)} users assigned. "
            f"Run: python manage.py runserver"
        ))
