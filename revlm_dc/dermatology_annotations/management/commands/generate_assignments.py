"""(Re)generate RCT assignments using defaults from assignments.py.

Processes users in ``registered_at`` order and simulates lesion counts
in-memory so the result is fully deterministic and reproducible.

Usage:
    python manage.py generate_assignments                       # all derm users
    python manage.py generate_assignments --role PCP            # all PCP users
    python manage.py generate_assignments --role both           # both roles
    python manage.py generate_assignments --users test          # specific user(s)
    python manage.py generate_assignments --dry-run             # preview only
"""

from django.core.management.base import BaseCommand

from dermatology_annotations.assignments import (
    DEFAULT_SEED,
    DEFAULT_ENABLED_FACTORS,
    ROLE_CONFIG,
    get_eligible_lesions,
    regenerate_all_assignments,
)
from dermatology_annotations.models import Dermatologist, PCPUser


ROLE_MODELS = {
    "Dermatologist": Dermatologist,
    "PCP": PCPUser,
}


class Command(BaseCommand):
    help = "Generate assignments for all (or specific) users using current defaults."

    def add_arguments(self, parser):
        parser.add_argument(
            "--users",
            nargs="+",
            help="Only assign these login IDs (default: all users in DB).",
        )
        parser.add_argument(
            "--role",
            choices=["Dermatologist", "PCP", "both"],
            default="Dermatologist",
            help="Which role to generate assignments for (default: Dermatologist).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Show what would happen without writing to the DB.",
        )

    def _run_for_role(self, role, eligible, options):
        cfg = ROLE_CONFIG[role]
        UserModel = ROLE_MODELS[role]

        self.stdout.write(self.style.MIGRATE_HEADING(f"\n--- {role} ---"))
        self.stdout.write(
            f"Settings: seed={DEFAULT_SEED}  "
            f"per_user={cfg['lesions_per_user']}  "
            f"overlap=[{cfg['min_annotators']},{cfg['max_annotators']}]  "
            f"enabled_factors={DEFAULT_ENABLED_FACTORS or '(none)'}"
        )

        if options["users"]:
            evaluators = UserModel.objects.filter(login_id__in=options["users"])
            missing = set(options["users"]) - set(evaluators.values_list("login_id", flat=True))
            if missing:
                self.stdout.write(self.style.WARNING(f"Users not found: {sorted(missing)}"))
        else:
            evaluators = UserModel.objects.all()

        if not evaluators.exists():
            self.stdout.write(self.style.WARNING("No users to assign."))
            return 0, 0

        evaluators = evaluators.order_by("registered_at", "login_id")
        dry = options["dry_run"]
        if dry:
            self.stdout.write(self.style.WARNING("DRY RUN — no changes will be written."))

        total = 0
        for evaluator, case_list in regenerate_all_assignments(
            evaluators, eligible, dry_run=dry, role=role,
        ):
            n = len(case_list)
            self.stdout.write(f"  {evaluator.login_id}: {n} assignments")
            total += n

        user_count = evaluators.count()
        self.stdout.write(self.style.SUCCESS(
            f"{'Would assign' if dry else 'Assigned'} {total} cases "
            f"across {user_count} {role} users."
        ))
        return total, user_count

    def handle(self, *args, **options):
        eligible = get_eligible_lesions()
        self.stdout.write(f"Eligible lesions: {len(eligible)}")
        if not eligible:
            self.stdout.write(self.style.ERROR("No eligible lesions. Aborting."))
            return

        role = options["role"]
        roles = ["Dermatologist", "PCP"] if role == "both" else [role]

        grand_total = 0
        grand_users = 0
        for r in roles:
            t, u = self._run_for_role(r, eligible, options)
            grand_total += t
            grand_users += u

        if len(roles) > 1:
            dry = options["dry_run"]
            self.stdout.write(self.style.SUCCESS(
                f"\nTotal: {'would assign' if dry else 'assigned'} {grand_total} cases "
                f"across {grand_users} users (all roles)."
            ))
