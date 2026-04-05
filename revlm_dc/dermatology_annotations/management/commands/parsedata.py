"""Django management command to convert prediction CSVs into the
annotations_data.json consumed by the dermatology review interface.

Usage:
    cd /scratch/jq2uw/derm_vlms/revlm_dc
    python manage.py parsedata
"""

import json
import os
import shutil
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

PROJECT_ROOT = Path(settings.BASE_DIR).parent

from dermatology_annotations.parse import parse_response

VLMS = {
    "MedGemma": "results/medgemma_predictions_all.csv",
    "GPT-5.3": "results/gpt53_predictions_all.csv",
    "DermatoLlama": "results/dermato_llama_predictions_all.csv",
}


class Command(BaseCommand):
    help = "Parse prediction CSVs into annotations_data.json and copy images."

    def handle(self, *args, **options):
        data_dir = Path(settings.BASE_DIR) / "data"
        image_dst = Path(settings.MEDIA_ROOT)
        image_src = PROJECT_ROOT / "results" / "images"

        data_dir.mkdir(parents=True, exist_ok=True)
        image_dst.mkdir(parents=True, exist_ok=True)

        # -- Load CSVs, keep only combined images --
        vlm_frames = {}
        for name, rel_path in VLMS.items():
            csv_path = PROJECT_ROOT / rel_path
            if not csv_path.exists():
                self.stdout.write(self.style.WARNING(
                    f"[SKIP] {name}: {csv_path} not found (predictions may still be running)"
                ))
                continue
            df = pd.read_csv(csv_path)
            df = df[df["id"].str.endswith("_combined")]
            if df.empty:
                self.stdout.write(self.style.WARNING(f"[SKIP] {name}: no combined rows"))
                continue
            vlm_frames[name] = df.set_index("id")
            self.stdout.write(f"  {name}: {len(df)} combined cases loaded")

        if not vlm_frames:
            self.stdout.write(self.style.ERROR("No prediction data found. Aborting."))
            return

        # -- Collect all combined case IDs across models --
        all_ids = sorted(set().union(*(df.index for df in vlm_frames.values())))
        self.stdout.write(f"\n{len(all_ids)} unique combined cases across {len(vlm_frames)} models")

        # -- Build annotations_data.json --
        annotations = {}
        parse_stats = {name: {"matched": 0, "missed": 0} for name in vlm_frames}

        for case_id in all_ids:
            case = {"image_path": f"{case_id}.jpg"}

            for model_name, df in vlm_frames.items():
                if case_id not in df.index:
                    continue
                raw = df.loc[case_id, "describe_then_classify"]
                if pd.isna(raw):
                    raw = ""
                descriptions, diagnoses = parse_response(str(raw))

                if diagnoses:
                    parse_stats[model_name]["matched"] += 1
                else:
                    parse_stats[model_name]["missed"] += 1

                case[model_name] = {
                    "diagnoses": diagnoses,
                    "descriptions": descriptions,
                }

            annotations[case_id] = case

        out_path = data_dir / "annotations_data.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        self.stdout.write(self.style.SUCCESS(f"\nWrote {out_path} ({len(annotations)} cases)"))

        # -- Parse statistics --
        self.stdout.write("\nParse statistics:")
        for name, stats in parse_stats.items():
            total = stats["matched"] + stats["missed"]
            self.stdout.write(
                f"  {name:20s}  diagnoses found: {stats['matched']}/{total}"
                f"  missed: {stats['missed']}"
            )

        # -- Placeholder users.json if missing --
        users_path = data_dir / "users.json"
        if not users_path.exists():
            placeholder = {"users": {"admin": list(all_ids)}}
            with open(users_path, "w", encoding="utf-8") as f:
                json.dump(placeholder, f, indent=2)
            self.stdout.write(self.style.SUCCESS(f"Wrote placeholder {users_path}"))
        else:
            self.stdout.write(f"users.json already exists, skipping ({users_path})")

        # -- Copy combined images --
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
        self.stdout.write(self.style.SUCCESS("\nDone. Ready to run: python manage.py runserver"))
