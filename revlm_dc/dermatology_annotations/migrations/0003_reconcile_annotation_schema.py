from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("dermatology_annotations", "0002_alter_annotation_id_alter_dermatologist_id"),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
DO $$
BEGIN
    -- Add columns expected by the current Django model, preserving old columns for compatibility.
    ALTER TABLE public.dermatology_annotations_annotation
        ADD COLUMN IF NOT EXISTS model varchar(100) NOT NULL DEFAULT '';

    ALTER TABLE public.dermatology_annotations_annotation
        ADD COLUMN IF NOT EXISTS interface_type varchar(20) NOT NULL DEFAULT 'conditional';

    ALTER TABLE public.dermatology_annotations_annotation
        ADD COLUMN IF NOT EXISTS raw_response text NOT NULL DEFAULT '';

    ALTER TABLE public.dermatology_annotations_annotation
        ADD COLUMN IF NOT EXISTS diagnosis_feedback jsonb NOT NULL DEFAULT '[]'::jsonb;

    ALTER TABLE public.dermatology_annotations_annotation
        ADD COLUMN IF NOT EXISTS description_feedback jsonb NOT NULL DEFAULT '[]'::jsonb;

    ALTER TABLE public.dermatology_annotations_annotation
        ADD COLUMN IF NOT EXISTS other_feedback jsonb NOT NULL DEFAULT '{"text": "", "crops": []}'::jsonb;

    ALTER TABLE public.dermatology_annotations_annotation
        ADD COLUMN IF NOT EXISTS user_diagnosis_1 jsonb NOT NULL DEFAULT '{"text": "", "crops": []}'::jsonb;

    ALTER TABLE public.dermatology_annotations_annotation
        ADD COLUMN IF NOT EXISTS user_diagnosis_2 jsonb NOT NULL DEFAULT '{"text": "", "crops": []}'::jsonb;

    ALTER TABLE public.dermatology_annotations_annotation
        ADD COLUMN IF NOT EXISTS user_diagnosis_3 jsonb NOT NULL DEFAULT '{"text": "", "crops": []}'::jsonb;

    ALTER TABLE public.dermatology_annotations_annotation
        ADD COLUMN IF NOT EXISTS user_reasons jsonb NOT NULL DEFAULT '{"text": "", "crops": []}'::jsonb;

    -- Replace old uniqueness (dermatologist, case_id) with model-aware uniqueness.
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'unique_dermatologist_case'
          AND conrelid = 'public.dermatology_annotations_annotation'::regclass
    ) THEN
        ALTER TABLE public.dermatology_annotations_annotation
            DROP CONSTRAINT unique_dermatologist_case;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'unique_dermatologist_case_model'
          AND conrelid = 'public.dermatology_annotations_annotation'::regclass
    ) THEN
        ALTER TABLE public.dermatology_annotations_annotation
            ADD CONSTRAINT unique_dermatologist_case_model
            UNIQUE (dermatologist_id, case_id, model);
    END IF;
END
$$;
            """,
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]
