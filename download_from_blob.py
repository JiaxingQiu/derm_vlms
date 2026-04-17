"""Download a blob prefix from Azure Blob Storage into a local folder.

Usage:
    python download_from_blob.py
    python download_from_blob.py configs/blob_config.yaml
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
        description="Download files from Azure Blob Storage while recreating folder structure locally."
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
                "Missing container name. Provide download.container_name when using an account-level azure.sas_url."
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


def normalize_prefix(prefix: str) -> str:
    cleaned = prefix.strip("/")
    return f"{cleaned}/" if cleaned else ""


def relative_blob_name(blob_name: str, blob_prefix: str) -> str:
    if blob_prefix and blob_name.startswith(blob_prefix):
        return blob_name[len(blob_prefix):]
    return blob_name


def download_folder(config: dict[str, Any]) -> None:
    download = config.get("download", {})
    if not isinstance(download, dict):
        raise ValueError("'download' must be a mapping in the config file.")

    container_name = download.get("container_name")
    blob_prefix = normalize_prefix(download.get("blob_prefix", ""))
    target_dir = download.get("target_dir")
    overwrite = bool(download.get("overwrite", False))

    if not target_dir:
        raise ValueError("Missing download.target_dir in config.")

    target_path = Path(target_dir).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    container_client = build_container_client(config, container_name)
    resolved_container_name = container_name or extract_container_name(container_client.url)

    downloaded_count = 0
    blobs = container_client.list_blobs(name_starts_with=blob_prefix)
    for blob in blobs:
        relative_name = relative_blob_name(blob.name, blob_prefix)
        if not relative_name or relative_name.endswith("/"):
            continue

        destination = target_path / Path(relative_name)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not overwrite:
            print(f"Skipping existing file {destination}")
            continue

        with open(destination, "wb") as handle:
            stream = container_client.download_blob(blob.name)
            handle.write(stream.readall())

        downloaded_count += 1
        print(f"Downloaded {resolved_container_name}/{blob.name} -> {destination}")

    print(
        f"Finished download: {downloaded_count} files from container "
        f"'{resolved_container_name}' with prefix '{blob_prefix}' into {target_path}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    download_folder(config)


if __name__ == "__main__":
    main()
