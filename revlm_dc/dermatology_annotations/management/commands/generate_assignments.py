"""(Re)generate RCT assignments using defaults from assignments.py.

Processes users in ``registered_at`` order and simulates lesion counts
in-memory so the result is fully deterministic and reproducible.

Usage:
    python manage.py generate_assignments              # all users in DB
    python manage.py generate_assignments --users test # specific user(s)
    python manage.py generate_assignments --dry-run    # preview only
"""

from django.core.management.base import BaseCommand

from dermatology_annotations.assignments import (
    DEFAULT_SEED,
    LESIONS_PER_USER,
    MIN_ANNOTATORS,
    MAX_ANNOTATORS,
    DEFAULT_ENABLED_FACTORS,
    get_eligible_lesions,
    regenerate_all_assignments,
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
            f"per_user={LESIONS_PER_USER}  "
            f"overlap=[{MIN_ANNOTATORS},{MAX_ANNOTATORS}]  "
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

        # Canonical order: registration time, then login_id as tiebreaker
        evaluators = evaluators.order_by("registered_at", "login_id")

        dry = options["dry_run"]
        if dry:
            self.stdout.write(self.style.WARNING("DRY RUN — no changes will be written.\n"))

        total = 0
        for evaluator, case_list in regenerate_all_assignments(
            evaluators, eligible, dry_run=dry,
        ):
            n = len(case_list)
            self.stdout.write(f"  {evaluator.login_id}: {n} assignments")
            total += n

        self.stdout.write(self.style.SUCCESS(
            f"\n{'Would assign' if dry else 'Assigned'} {total} cases "
            f"across {evaluators.count()} users."
        ))
