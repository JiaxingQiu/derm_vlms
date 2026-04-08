"""Upload a local folder tree to Azure Blob Storage.

Usage:
    python upload_to_blob.py
    python upload_to_blob.py configs/blob_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

import yaml
from azure.storage.blob import ContainerClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local folder to Azure Blob Storage while preserving its structure."
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


def build_container_url(sas_url: str, sas_token: str | None) -> str:
    if "?" in sas_url:
        return sas_url
    if sas_token:
        return f"{sas_url.rstrip('?')}?{sas_token.lstrip('?')}"
    return sas_url


def build_container_client(config: dict[str, Any], container_name: str | None) -> ContainerClient:
    azure = config.get("azure", {})
    if not isinstance(azure, dict):
        raise ValueError("'azure' must be a mapping in the config file.")

    sas_url = azure.get("sas_url")
    sas_token = azure.get("sas_token")

    if sas_url:
        return ContainerClient.from_container_url(build_container_url(sas_url, sas_token))

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


def upload_folder(config: dict[str, Any]) -> None:
    upload = config.get("upload", {})
    if not isinstance(upload, dict):
        raise ValueError("'upload' must be a mapping in the config file.")

    source_dir = upload.get("source_dir")
    container_name = upload.get("container_name")
    blob_prefix = normalize_prefix(upload.get("blob_prefix", ""))
    overwrite = bool(upload.get("overwrite", False))

    if not source_dir:
        raise ValueError("Missing upload.source_dir in config.")
    source_path = Path(source_dir).expanduser().resolve()
    if not source_path.exists() or not source_path.is_dir():
        raise ValueError(f"Source directory does not exist: {source_path}")

    container_client = build_container_client(config, container_name)
    resolved_container_name = container_name or extract_container_name(container_client.url)

    uploaded_count = 0
    for file_path in iter_files(source_path):
        relative_path = file_path.relative_to(source_path).as_posix()
        blob_name = f"{blob_prefix}{relative_path}"
        blob_client = container_client.get_blob_client(blob_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)

        uploaded_count += 1
        print(f"Uploaded {file_path} -> {resolved_container_name}/{blob_name}")

    print(
        f"Finished upload: {uploaded_count} files from {source_path} "
        f"to container '{resolved_container_name}' with prefix '{blob_prefix}'"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    upload_folder(config)


if __name__ == "__main__":
    main()
