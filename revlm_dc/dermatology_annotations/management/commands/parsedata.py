"""Django management command to convert prediction CSVs into the
annotations_data.json consumed by the dermatology review interface.

Includes all standard image modes (photo, dscope, combined) per lesion.
The ``virtual`` mode is excluded by default (use --include-virtual to add it).

Usage:
    cd /scratch/jq2uw/derm_vlms/revlm_dc
    python manage.py parsedata
    python manage.py parsedata --modes photo dscope combined
    python manage.py parsedata --include-virtual
"""

import json
import shutil
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

PROJECT_ROOT = Path(settings.BASE_DIR).parent

from dermatology_annotations.parse import parse_reason_response

VLMS = {
    "MedGemma": "results/medgemma_predictions_reason.csv",
    "GPT-5.3": "results/gpt53_predictions_reason.csv",
    "DermatoLlama": "results/dermato_llama_predictions_reason.csv",
}

RESPONSE_COL = "reason_classify"

STANDARD_MODES = {"photo", "dscope", "combined"}


class Command(BaseCommand):
    help = "Parse prediction CSVs into annotations_data.json and copy images."

    def add_arguments(self, parser):
        parser.add_argument(
            "--modes",
            nargs="+",
            default=list(STANDARD_MODES),
            help="Image modes to include (default: photo dscope combined).",
        )
        parser.add_argument(
            "--include-virtual",
            action="store_true",
            default=False,
            help="Also include the 'virtual' image mode.",
        )

    def handle(self, *args, **options):
        allowed_modes = set(options["modes"])
        if options["include_virtual"]:
            allowed_modes.add("virtual")
        self.stdout.write(f"Including image modes: {sorted(allowed_modes)}")

        data_dir = Path(settings.BASE_DIR) / "data"
        image_dst = Path(settings.MEDIA_ROOT)
        image_src = PROJECT_ROOT / "results" / "images"

        data_dir.mkdir(parents=True, exist_ok=True)
        image_dst.mkdir(parents=True, exist_ok=True)

        vlm_frames = {}
        for name, rel_path in VLMS.items():
            csv_path = PROJECT_ROOT / rel_path
            if not csv_path.exists():
                self.stdout.write(self.style.WARNING(
                    f"[SKIP] {name}: {csv_path} not found"
                ))
                continue
            df = pd.read_csv(csv_path)
            df = df[df["image_mode"].isin(allowed_modes)]
            if df.empty:
                self.stdout.write(self.style.WARNING(
                    f"[SKIP] {name}: no rows for modes {sorted(allowed_modes)}"
                ))
                continue
            vlm_frames[name] = df.set_index("id")
            per_mode = df["image_mode"].value_counts().to_dict()
            self.stdout.write(f"  {name}: {len(df)} rows  {per_mode}")

        if not vlm_frames:
            self.stdout.write(self.style.ERROR("No prediction data found. Aborting."))
            return

        all_ids = sorted(set().union(*(df.index for df in vlm_frames.values())))
        self.stdout.write(
            f"\n{len(all_ids)} unique case IDs across {len(vlm_frames)} models"
        )

        annotations = {}
        parse_stats = {name: {"matched": 0, "missed": 0} for name in vlm_frames}

        for case_id in all_ids:
            case = {"image_path": f"{case_id}.jpg"}

            for model_name, df in vlm_frames.items():
                if case_id not in df.index:
                    continue
                raw = df.loc[case_id, RESPONSE_COL]
                if pd.isna(raw):
                    raw = ""
                diagnoses = parse_reason_response(str(raw))

                if diagnoses:
                    parse_stats[model_name]["matched"] += 1
                else:
                    parse_stats[model_name]["missed"] += 1

                case[model_name] = {
                    "raw_response": str(raw),
                    "diagnoses": diagnoses,
                }

            annotations[case_id] = case

        out_path = data_dir / "annotations_data.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        self.stdout.write(self.style.SUCCESS(
            f"\nWrote {out_path} ({len(annotations)} cases)"
        ))

        self.stdout.write("\nParse statistics:")
        for name, stats in parse_stats.items():
            total = stats["matched"] + stats["missed"]
            self.stdout.write(
                f"  {name:20s}  diagnoses found: {stats['matched']}/{total}"
                f"  missed: {stats['missed']}"
            )

        copied = 0
        skipped = 0
        for case_id in all_ids:
            src = image_src / f"{case_id}.jpg"
            dst = image_dst / f"{case_id}.jpg"
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                skipped += 1
                self.stdout.write(self.style.WARNING(f"  Image not found: {src}"))

        self.stdout.write(self.style.SUCCESS(
            f"\nCopied {copied} images to {image_dst}"
            + (f" ({skipped} missing)" if skipped else "")
        ))
        self.stdout.write(self.style.SUCCESS(
            "\nDone. Run: python manage.py generate_assignments"
        ))
