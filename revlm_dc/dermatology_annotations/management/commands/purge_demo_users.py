"""Delete the throwaway demo_* evaluators left behind by the /demo/ link.

/demo/ already collects finished visitors on each visit, but a quiet site
never triggers that, so abandoned rows can sit around indefinitely. Run this
on a schedule instead, e.g. every half hour:

    */30 * * * * cd /path/to/revlm_dc && python manage.py purge_demo_users

Usage:
    python manage.py purge_demo_users              # collect finished visitors
    python manage.py purge_demo_users --dry-run    # list them, delete nothing
    python manage.py purge_demo_users --all        # also evict active visitors

Every mode selects rows through ``_demo_evaluator_qs``, the same queryset the
live site uses, which requires a ``demo_`` login id *and* a null
``years_experience`` *and* a null ``assignment_slot``. Registered users always
have the latter two set, so no real account can be reached from here.
"""

from django.core.management.base import BaseCommand

from dermatology_annotations.views import (
    _demo_evaluator_qs,
    finished_demo_evaluators,
)


class Command(BaseCommand):
    help = "Delete demo_* evaluators whose sessions have ended."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List what would be removed without touching the database.",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Also remove demo evaluators who are still mid-session.",
        )

    def handle(self, *args, **options):
        qs = _demo_evaluator_qs() if options["all"] else finished_demo_evaluators()
        login_ids = sorted(qs.values_list("login_id", flat=True))

        if not login_ids:
            self.stdout.write("No demo evaluators to remove.")
            return

        for login_id in login_ids:
            self.stdout.write(f"  {login_id}")

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING(
                f"DRY RUN — would remove {len(login_ids)} demo evaluator(s)."
            ))
            return

        _total, per_model = qs.delete()
        self.stdout.write(self.style.SUCCESS(
            f"Removed {per_model.get('dermatology_annotations.Dermatologist', 0)} "
            f"demo evaluator(s), "
            f"{per_model.get('dermatology_annotations.Assignment', 0)} assignment(s), "
            f"{per_model.get('dermatology_annotations.Annotation', 0)} annotation(s)."
        ))
