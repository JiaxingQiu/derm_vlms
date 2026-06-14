"""Shared paths and a small module loader for the simedit pipeline.

Everything the simulation writes stays under ``prelim_simedit/results_local``.
The only outside reads are ``data_share/*.parquet`` (ground truth + mapping)
and the source lesion images in ``data/``.
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATA_SHARE = PROJECT_ROOT / "data_share"
MIDAS_SHARE = DATA_SHARE / "midas_share.parquet"
CASE_MAPPING = DATA_SHARE / "case_mapping.parquet"

SIMEDIT_ROOT = PROJECT_ROOT / "prelim_simedit"
RESULTS_LOCAL = SIMEDIT_ROOT / "results_local"

# Make the repo root importable (for `prelim_acc`, `tokens`, etc.)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_module_from_path(name, path):
    """Load a standalone .py file as a uniquely-named module.

    Used to reuse the existing ``collect_ai_response/<model>/utils.py`` loaders
    without import-name clashes (several of them are just named ``utils``).
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
