"""Upload a local folder tree to Azure Blob Storage.

Usage:
    python upload_to_blob.py
    python upload_to_blob.py configs/blob_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from azure.storage.blob import BlobServiceClient


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


def build_service_client(config: dict[str, Any]) -> BlobServiceClient:
    azure = config.get("azure", {})
    if not isinstance(azure, dict):
        raise ValueError("'azure' must be a mapping in the config file.")

    connection_string = azure.get("connection_string")
    account_url = azure.get("account_url")
    credential = azure.get("credential")

    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)
    if account_url:
        return BlobServiceClient(account_url=account_url, credential=credential)

    raise ValueError(
        "Provide either azure.connection_string or azure.account_url in the config."
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
    if not container_name:
        raise ValueError("Missing upload.container_name in config.")

    source_path = Path(source_dir).expanduser().resolve()
    if not source_path.exists() or not source_path.is_dir():
        raise ValueError(f"Source directory does not exist: {source_path}")

    service_client = build_service_client(config)
    container_client = service_client.get_container_client(container_name)

    uploaded_count = 0
    for file_path in iter_files(source_path):
        relative_path = file_path.relative_to(source_path).as_posix()
        blob_name = f"{blob_prefix}{relative_path}"
        blob_client = container_client.get_blob_client(blob_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)

        uploaded_count += 1
        print(f"Uploaded {file_path} -> {container_name}/{blob_name}")

    print(
        f"Finished upload: {uploaded_count} files from {source_path} "
        f"to container '{container_name}' with prefix '{blob_prefix}'"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    upload_folder(config)


if __name__ == "__main__":
    main()
