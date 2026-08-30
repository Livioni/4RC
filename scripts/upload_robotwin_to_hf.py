#!/usr/bin/env python3
"""Archive RoboTwin by task and upload it to a Hugging Face dataset repo.

The source contains millions of small files, so this script creates one
uncompressed tar archive per task. Every archive stores paths relative to the
RoboTwin root, which means extracting all archives reconstructs the original
task/episode directory tree.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi


DEFAULT_SOURCE = Path("datasets/RoboTwin")
DEFAULT_REPO_ID = "HarrisonPENG/4RC-Action"
DEFAULT_REPO_PATH = "RoboTwin"
DEFAULT_STAGING_DIR = Path("outputs/hf-upload-robotwin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package RoboTwin into one tar per task and upload the archives "
            "to a Hugging Face dataset repository."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--path-in-repo",
        default=DEFAULT_REPO_PATH,
        help="Remote folder containing task tar archives.",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face write token. Falls back to the HF_TOKEN environment variable.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository if it does not exist.",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Upload only this task. May be supplied multiple times.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING_DIR,
        help="Temporary task archives are written here.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep local tar archives after a successful upload.",
    )
    parser.add_argument(
        "--rebuild-archives",
        action="store_true",
        help="Rebuild an archive even when it already exists in the staging directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Upload tasks even when their remote archive already exists.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--skip-layout-doc",
        action="store_true",
        help="Do not upload RoboTwin/README.md describing the archive layout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the selected tasks without uploading.",
    )
    return parser.parse_args()


def normalize_repo_path(value: str) -> str:
    normalized = str(PurePosixPath(value.strip("/")))
    if normalized in ("", "."):
        return ""
    if normalized.startswith("../") or normalized == "..":
        raise ValueError("path-in-repo must stay inside the repository")
    return normalized


def list_tasks(source: Path, selected: list[str] | None) -> list[Path]:
    available = {path.name: path for path in source.iterdir() if path.is_dir()}
    if not available:
        raise RuntimeError(f"No task directories found below {source}")
    if selected is None:
        return [available[name] for name in sorted(available)]

    requested = list(dict.fromkeys(selected))
    missing = sorted(set(requested).difference(available))
    if missing:
        raise ValueError(f"Unknown RoboTwin task(s): {missing}")
    return [available[name] for name in requested]


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def scan_task(task_dir: Path) -> tuple[int, int]:
    total_bytes = 0
    file_count = 0
    for directory, _, filenames in os.walk(task_dir):
        for filename in filenames:
            path = Path(directory, filename)
            try:
                total_bytes += path.stat().st_size
            except FileNotFoundError:
                raise RuntimeError(f"File disappeared while scanning: {path}") from None
            file_count += 1
    return total_bytes, file_count


def normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def build_task_archive(
    task_dir: Path,
    archive_path: Path,
    *,
    rebuild: bool,
) -> Path:
    if archive_path.is_file() and not rebuild:
        print(f"[archive] Reusing {archive_path}")
        return archive_path

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    source_bytes, file_count = scan_task(task_dir)
    free_bytes = shutil.disk_usage(archive_path.parent).free
    reserve = max(512 * 1024**2, source_bytes // 20)
    required = source_bytes + reserve
    if free_bytes < required:
        raise RuntimeError(
            f"Not enough staging space for {task_dir.name}: "
            f"need about {format_bytes(required)}, have {format_bytes(free_bytes)}. "
            "Use --staging-dir on a larger disk."
        )

    partial_path = Path(f"{archive_path}.partial")
    if partial_path.exists():
        partial_path.unlink()
    print(
        f"[archive] Building {archive_path.name}: "
        f"{file_count:,} files, {format_bytes(source_bytes)}"
    )
    try:
        with tarfile.open(partial_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
            archive.add(
                task_dir,
                arcname=task_dir.name,
                recursive=True,
                filter=normalized_tar_info,
            )
        partial_path.replace(archive_path)
    except BaseException:
        print(
            f"[archive] Interrupted archive retained at {partial_path}",
            file=sys.stderr,
        )
        raise
    print(f"[archive] Ready: {archive_path} ({format_bytes(archive_path.stat().st_size)})")
    return archive_path


def upload_with_retries(
    api: HfApi,
    *,
    archive_path: Path,
    remote_path: str,
    repo_id: str,
    token: str,
    revision: str | None,
    retries: int,
) -> Any:
    if retries < 1:
        raise ValueError("--retries must be at least 1")
    for attempt in range(1, retries + 1):
        try:
            return api.upload_file(
                path_or_fileobj=archive_path,
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                token=token,
                commit_message=f"Upload RoboTwin task archive: {archive_path.stem}",
            )
        except Exception as error:
            if attempt == retries:
                raise
            delay = min(60, 5 * 2 ** (attempt - 1))
            print(
                f"[upload] Attempt {attempt}/{retries} failed: {error}. "
                f"Retrying in {delay}s.",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def layout_document(repo_id: str, repo_path: str) -> bytes:
    prefix = f"{repo_path}/" if repo_path else ""
    text = f"""# RoboTwin archive layout

This directory contains the RoboTwin data used by 4RC.

Each task is stored as one uncompressed tar archive:

    {prefix}<task>.tar

Every archive contains:

    <task>/<episode>/images/...
    <task>/<episode>/depths/...
    <task>/<episode>/intrinsics/...
    <task>/<episode>/extrinsics/...

Download and reconstruct the original RoboTwin directory:

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id="{repo_id}",
        repo_type="dataset",
        allow_patterns="{prefix}*.tar",
    )

Then extract all tar files into the same destination directory. The archives
are not compressed because the source images are already PNG-compressed and
uncompressed tar supports faster creation and extraction.
"""
    return text.encode("utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"RoboTwin source does not exist: {source}")
    repo_path = normalize_repo_path(args.path_in_repo)
    staging_dir = args.staging_dir.expanduser().resolve()
    if source == staging_dir or source in staging_dir.parents:
        raise ValueError("staging-dir must not be inside the RoboTwin source")
    tasks = list_tasks(source, args.tasks)

    print(f"Source: {source}")
    print(f"Destination: https://huggingface.co/datasets/{args.repo_id}")
    print(f"Remote path: {repo_path or '/'}")
    print(f"Selected tasks: {len(tasks)}")
    for task in tasks:
        print(f"  - {task.name}")
    if args.dry_run:
        print("Dry run complete; no repository or local archive was created.")
        return

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Pass --token or set the HF_TOKEN environment variable")

    api = HfApi(token=token)
    identity = api.whoami(token=token)
    print(f"Authenticated as: {identity.get('name', '<unknown>')}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=True if args.private else None,
        exist_ok=True,
        token=token,
    )
    staging_dir.mkdir(parents=True, exist_ok=True)

    for position, task_dir in enumerate(tasks, start=1):
        remote_path = str(PurePosixPath(repo_path, f"{task_dir.name}.tar"))
        print(f"[{position}/{len(tasks)}] {task_dir.name} -> {remote_path}")
        if not args.overwrite and api.file_exists(
            repo_id=args.repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=args.revision,
            token=token,
        ):
            print("[upload] Remote archive already exists; skipping.")
            continue

        archive_path = staging_dir / f"{task_dir.name}.tar"
        build_task_archive(
            task_dir,
            archive_path,
            rebuild=args.rebuild_archives,
        )
        commit = upload_with_retries(
            api,
            archive_path=archive_path,
            remote_path=remote_path,
            repo_id=args.repo_id,
            token=token,
            revision=args.revision,
            retries=args.retries,
        )
        print(f"[upload] Complete: {getattr(commit, 'commit_url', commit)}")
        if not args.keep_archives:
            archive_path.unlink()
            print(f"[archive] Removed uploaded staging file: {archive_path}")

    if not args.skip_layout_doc:
        readme_path = str(PurePosixPath(repo_path, "README.md"))
        api.upload_file(
            path_or_fileobj=layout_document(args.repo_id, repo_path),
            path_in_repo=readme_path,
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            token=token,
            commit_message="Document RoboTwin archive layout",
        )
        print(f"[upload] Layout documentation updated: {readme_path}")

    print("All selected RoboTwin tasks are uploaded.")


if __name__ == "__main__":
    main()
