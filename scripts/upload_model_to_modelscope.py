#!/usr/bin/env python3
"""Upload a model checkpoint (single file or directory) to a ModelScope model repository."""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_MODEL_PATH = Path("outputs/4rc-stage2-action-bs4x3/checkpoint-90000")
DEFAULT_REPO_ID = "livion/4RC-Action-RoboTwin-Stage2"
DEFAULT_ENDPOINT = "https://modelscope.cn"
DEFAULT_EXCLUDES = ("optimizer*", "scheduler*", "random_states*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a model checkpoint (file or directory) to a ModelScope model repository."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local model file or checkpoint directory (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Destination repository in owner/name form (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Destination path in the repository (default: local file name; for a "
        "directory upload it is used as a prefix for the relative file paths).",
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
        "--exclude",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Extra glob pattern for file names to skip in a directory upload (repeatable).",
    )
    parser.add_argument(
        "--keep-training-state",
        action="store_true",
        help="Also upload optimizer/scheduler/random-state files from a checkpoint directory.",
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


def collect_upload_plan(
    model_path: Path, exclude_patterns: list[str]
) -> tuple[list[Path], list[Path]]:
    if model_path.is_file():
        return [model_path], []
    upload: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(p for p in model_path.rglob("*") if p.is_file()):
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in exclude_patterns):
            skipped.append(path)
        else:
            upload.append(path)
    return upload, skipped


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
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not args.ms_token.strip():
        raise ValueError("--ms-token cannot be empty")

    repo_id = normalize_repo_id(args.repo_id)

    exclude_patterns: list[str] = list(args.exclude or [])
    if model_path.is_dir() and not args.keep_training_state:
        exclude_patterns = list(DEFAULT_EXCLUDES) + exclude_patterns
    upload_files, skipped_files = collect_upload_plan(model_path, exclude_patterns)
    if not upload_files:
        raise ValueError(f"Nothing to upload under {model_path}")

    prefix = normalize_repo_path(args.path_in_repo, model_path) if args.path_in_repo else ""
    plan: list[tuple[Path, str]] = []
    for file_path in upload_files:
        if model_path.is_dir():
            relative = file_path.relative_to(model_path).as_posix()
            remote = f"{prefix}/{relative}" if prefix else relative
        else:
            remote = normalize_repo_path(args.path_in_repo, file_path)
        plan.append((file_path, normalize_repo_path(remote, file_path)))

    total_bytes = sum(file_path.stat().st_size for file_path, _ in plan)
    print(f"Destination: {args.endpoint.rstrip('/')}/models/{repo_id}")
    print(f"Revision: {args.revision}")
    print(f"Files to upload: {len(plan)} ({format_bytes(total_bytes)} total)")
    for file_path, remote in plan:
        print(f"  {file_path} ({format_bytes(file_path.stat().st_size)}) -> {remote}")
    for file_path in skipped_files:
        print(f"  skipped (training state): {file_path.relative_to(model_path)}")
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

    for file_path, remote in plan:
        commit_message = args.commit_message or f"Upload {remote}"
        result = api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=file_path,
            path_in_repo=remote,
            revision=args.revision,
            commit_message=commit_message,
        )
        print(f"Uploaded: {result_url(result, args.endpoint, repo_id, remote)}")
    print("Upload complete.")


if __name__ == "__main__":
    main()
