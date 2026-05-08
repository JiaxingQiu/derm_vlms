"""(Re)generate RCT assignments using defaults from assignments.py.

Reads DEFAULT_SEED, DEFAULT_MAX_LESIONS, and DEFAULT_ENABLED_FACTORS
directly — no YAML config file needed.  Existing assignments for each
user are replaced.

Usage:
    python manage.py generate_assignments              # all users in DB
    python manage.py generate_assignments --users test # specific user(s)
    python manage.py generate_assignments --dry-run    # preview only
"""

from django.core.management.base import BaseCommand

from dermatology_annotations.assignments import (
    DEFAULT_SEED,
    DEFAULT_MAX_LESIONS,
    DEFAULT_ENABLED_FACTORS,
    assign_cases_for_user,
    build_case_list_for_user,
    get_eligible_lesions,
)
from dermatology_annotations.models import Dermatologist


class Command(BaseCommand):
    help = "Generate assignments for all (or specific) users using current defaults."

    def add_arguments(self, parser):
        parser.add_argument(
            "--users",
            nargs="+",
            help="Only assign these login IDs (default: all users in DB).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Show what would happen without writing to the DB.",
        )

    def handle(self, *args, **options):
        self.stdout.write(
            f"Settings: seed={DEFAULT_SEED}  "
            f"max_lesions={DEFAULT_MAX_LESIONS}  "
            f"enabled_factors={DEFAULT_ENABLED_FACTORS or '(none)'}"
        )

        eligible = get_eligible_lesions()
        self.stdout.write(f"Eligible lesions: {len(eligible)}")
        if not eligible:
            self.stdout.write(self.style.ERROR("No eligible lesions. Aborting."))
            return

        if options["users"]:
            evaluators = Dermatologist.objects.filter(login_id__in=options["users"])
            missing = set(options["users"]) - set(evaluators.values_list("login_id", flat=True))
            if missing:
                self.stdout.write(self.style.WARNING(f"Users not found: {sorted(missing)}"))
        else:
            evaluators = Dermatologist.objects.all()

        if not evaluators.exists():
            self.stdout.write(self.style.WARNING("No users to assign."))
            return

        dry = options["dry_run"]
        if dry:
            self.stdout.write(self.style.WARNING("DRY RUN — no changes will be written.\n"))

        total = 0
        for evaluator in evaluators.order_by("login_id"):
            if dry:
                case_list, _ = build_case_list_for_user(
                    evaluator.login_id, eligible,
                    DEFAULT_SEED, DEFAULT_MAX_LESIONS, DEFAULT_ENABLED_FACTORS,
                )
                n = len(case_list)
            else:
                assignments = assign_cases_for_user(evaluator, eligible)
                n = len(assignments)
            self.stdout.write(f"  {evaluator.login_id}: {n} assignments")
            total += n

        self.stdout.write(self.style.SUCCESS(
            f"\n{'Would assign' if dry else 'Assigned'} {total} cases "
            f"across {evaluators.count()} users."
        ))
