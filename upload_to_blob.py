"""Upload local folder trees to Azure Blob Storage.

Supports multiple upload specs via a ``uploads`` list in the YAML config,
or the legacy single ``upload`` key for backward compatibility.

Usage:
    python upload_to_blob.py
    python upload_to_blob.py configs/blob_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse, urlsplit, urlunsplit
from typing import Any

import yaml
from azure.storage.blob import ContainerClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload local folders to Azure Blob Storage while preserving structure."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=Path(__file__).resolve().parent / "configs" / "blob_config.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    return config


def extract_container_name(container_url: str) -> str:
    path_parts = urlparse(container_url).path.strip("/").split("/")
    if not path_parts or not path_parts[0]:
        raise ValueError(f"Could not extract container name from SAS URL: {container_url}")
    return path_parts[0]


def build_container_url(sas_url: str, sas_token: str | None, container_name: str | None) -> str:
    split = urlsplit(sas_url)
    query = sas_token.lstrip("?") if sas_token else split.query

    path_parts = [part for part in split.path.strip("/").split("/") if part]
    if path_parts:
        container_path = f"/{path_parts[0]}"
    else:
        if not container_name:
            raise ValueError(
                "Missing container name. Provide container_name when using an account-level azure.sas_url."
            )
        container_path = f"/{container_name.strip('/')}"

    return urlunsplit((split.scheme, split.netloc, container_path, query, split.fragment))


def build_container_client(config: dict[str, Any], container_name: str | None) -> ContainerClient:
    azure = config.get("azure", {})
    if not isinstance(azure, dict):
        raise ValueError("'azure' must be a mapping in the config file.")

    sas_url = azure.get("sas_url")
    sas_token = azure.get("sas_token")

    if sas_url:
        return ContainerClient.from_container_url(build_container_url(sas_url, sas_token, container_name))

    raise ValueError(
        "Provide azure.sas_url, optionally with azure.sas_token if the URL does not already include it."
    )


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def normalize_prefix(prefix: str) -> str:
    cleaned = prefix.strip("/")
    return f"{cleaned}/" if cleaned else ""


def get_upload_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a list of upload specs from either ``uploads`` (list) or ``upload`` (single)."""
    if "uploads" in config:
        specs = config["uploads"]
        if not isinstance(specs, list):
            raise ValueError("'uploads' must be a list in the config file.")
        return specs

    if "upload" in config:
        spec = config["upload"]
        if not isinstance(spec, dict):
            raise ValueError("'upload' must be a mapping in the config file.")
        return [spec]

    raise ValueError("Config must contain either 'uploads' (list) or 'upload' (dict).")


def upload_one(config: dict[str, Any], spec: dict[str, Any]) -> None:
    name = spec.get("name", "unnamed")
    source_dir = spec.get("source_dir")
    container_name = spec.get("container_name")
    blob_prefix = normalize_prefix(spec.get("blob_prefix", ""))
    overwrite = bool(spec.get("overwrite", False))

    if not source_dir:
        raise ValueError(f"Missing source_dir in upload spec '{name}'.")
    source_path = Path(source_dir).expanduser().resolve()
    if not source_path.exists() or not source_path.is_dir():
        print(f"[SKIP] {name}: source directory does not exist: {source_path}")
        return

    container_client = build_container_client(config, container_name)
    resolved_container_name = container_name or extract_container_name(container_client.url)

    uploaded_count = 0
    skipped_count = 0
    for file_path in iter_files(source_path):
        relative_path = file_path.relative_to(source_path).as_posix()
        blob_name = f"{blob_prefix}{relative_path}"
        blob_client = container_client.get_blob_client(blob_name)

        if not overwrite and blob_client.exists():
            skipped_count += 1
            continue

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)

        uploaded_count += 1
        print(f"  [{name}] Uploaded {file_path} -> {resolved_container_name}/{blob_name}")

    print(
        f"[{name}] {uploaded_count} uploaded, {skipped_count} skipped "
        f"({source_path} -> {resolved_container_name}/{blob_prefix})"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    specs = get_upload_specs(config)

    print(f"Found {len(specs)} upload spec(s).\n")
    for spec in specs:
        upload_one(config, spec)
        print()


if __name__ == "__main__":
    main()
