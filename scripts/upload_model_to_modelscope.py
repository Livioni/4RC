#!/usr/bin/env python3
"""Upload a single model checkpoint to a ModelScope model repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_MODEL_PATH = Path(
    "outputs/4rc-robotwin-geometry/checkpoint-5000/model.safetensors"
)
DEFAULT_REPO_ID = "livion/4RC-Geometry"
DEFAULT_ENDPOINT = "https://modelscope.cn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload one model checkpoint to a ModelScope model repository."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local model file (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Destination repository in owner/name form (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Destination file path in the repository (default: local file name).",
    )
    parser.add_argument(
        "--ms-token",
        required=True,
        help="ModelScope access token with write permission (required).",
    )
    parser.add_argument(
        "--revision",
        default="master",
        help="Destination branch (default: master).",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit message (default: derived from the destination file name).",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"ModelScope endpoint (default: {DEFAULT_ENDPOINT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the upload plan without connecting to ModelScope.",
    )
    return parser.parse_args()


def normalize_repo_id(repo_id: str) -> str:
    repo_id = repo_id.strip().strip("/")
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("--repo-id must use the owner/name form")
    return repo_id


def normalize_repo_path(path_in_repo: str | None, model_path: Path) -> str:
    raw_path = path_in_repo if path_in_repo is not None else model_path.name
    raw_path = raw_path.strip().replace("\\", "/")
    path = PurePosixPath(raw_path)
    if not raw_path or path.is_absolute() or ".." in path.parts:
        raise ValueError("--path-in-repo must be a relative path inside the repository")
    return str(path)


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def result_url(result: Any, endpoint: str, repo_id: str, path_in_repo: str) -> str:
    if isinstance(result, dict):
        for key in ("commit_url", "url", "Url"):
            if result.get(key):
                return str(result[key])
    for attribute in ("commit_url", "url"):
        value = getattr(result, attribute, None)
        if value:
            return str(value)
    return f"{endpoint.rstrip('/')}/models/{repo_id}/files/{path_in_repo}"


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if not args.ms_token.strip():
        raise ValueError("--ms-token cannot be empty")

    repo_id = normalize_repo_id(args.repo_id)
    path_in_repo = normalize_repo_path(args.path_in_repo, model_path)
    commit_message = args.commit_message or f"Upload {path_in_repo}"

    print(f"Source: {model_path} ({format_bytes(model_path.stat().st_size)})")
    print(f"Destination: {args.endpoint.rstrip('/')}/models/{repo_id}")
    print(f"Remote path: {path_in_repo}")
    print(f"Revision: {args.revision}")
    if args.dry_run:
        print("Dry run complete; nothing was uploaded.")
        return

    try:
        from modelscope_hub import HubApi
    except ImportError:
        print(
            "Missing dependency 'modelscope-hub'. Install project dependencies "
            "or run: pip install modelscope-hub",
            file=sys.stderr,
        )
        raise SystemExit(2) from None

    api = HubApi(token=args.ms_token, endpoint=args.endpoint)
    identity = api.whoami()
    username = getattr(identity, "username", None)
    if username:
        print(f"Authenticated as: {username}")

    result = api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=model_path,
        path_in_repo=path_in_repo,
        revision=args.revision,
        commit_message=commit_message,
    )
    print(f"Upload complete: {result_url(result, args.endpoint, repo_id, path_in_repo)}")


if __name__ == "__main__":
    main()
