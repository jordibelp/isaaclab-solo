#!/usr/bin/env python3
"""Download W&B run checkpoint files into the local SKRL checkpoint cache.

Default output name:
    %m%d_<run_id>_best_model.pt

Example:
    python source/scripts/skrl/download_wandb_best_checkpoint.py \
        5n2klf2x iqzxittl --date-prefix 0608
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
import tempfile
from pathlib import Path


DEFAULT_ENTITY = "jordibelp"
DEFAULT_PROJECT = "borinotIsaacLab"
DEFAULT_SOURCE_FILE = "best_model.pt"
DEFAULT_CHECKPOINT_DIR = Path("/home/jordibelp/IsaacLab/logs/skrl/checkpoints")


def _current_mmdd() -> str:
    return dt.datetime.now().strftime("%m%d")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download best_model.pt from W&B runs into logs/skrl/checkpoints."
    )
    parser.add_argument("run_ids", nargs="+", help="W&B run ids, e.g. 5n2klf2x iqzxittl")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help=f"W&B entity (default: {DEFAULT_ENTITY})")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"W&B project (default: {DEFAULT_PROJECT})")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Destination directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--source-file",
        default=DEFAULT_SOURCE_FILE,
        help=f"File to fetch from each run (default: {DEFAULT_SOURCE_FILE})",
    )
    parser.add_argument(
        "--date-prefix",
        default=_current_mmdd(),
        help="Filename prefix. Defaults to today's %%m%%d, e.g. 0508.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination files if they already exist.",
    )
    return parser.parse_args()


def _download_run_file(api, *, entity: str, project: str, run_id: str, source_file: str, destination: Path) -> None:
    run_path = f"{entity}/{project}/{run_id}"
    run = api.run(run_path)
    try:
        wandb_file = run.file(source_file)
    except Exception as exc:  # W&B raises CommError for missing files.
        available = sorted(file.name for file in run.files() if file.name.endswith(".pt"))
        preview = ", ".join(available[:20])
        if len(available) > 20:
            preview += f", ... ({len(available)} .pt files total)"
        raise RuntimeError(
            f"Run {run_path} does not expose {source_file!r}. Available .pt files: {preview or 'none'}"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"wandb-{run_id}-") as tmp_dir:
        downloaded_path = Path(wandb_file.download(root=tmp_dir, replace=True).name)
        shutil.copy2(downloaded_path, destination)


def main() -> int:
    args = _parse_args()

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed in this Python environment.") from exc

    api = wandb.Api(timeout=60)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()

    for run_id in args.run_ids:
        destination = checkpoint_dir / f"{args.date_prefix}_{run_id}_best_model.pt"
        if destination.exists() and not args.overwrite:
            print(f"SKIP {run_id}: {destination} already exists (use --overwrite to replace)")
            continue
        _download_run_file(
            api,
            entity=args.entity,
            project=args.project,
            run_id=run_id,
            source_file=args.source_file,
            destination=destination,
        )
        size_mib = destination.stat().st_size / (1024 * 1024)
        print(f"OK {run_id}: {destination} ({size_mib:.2f} MiB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
