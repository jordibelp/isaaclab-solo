import atexit
import inspect
import json
import re
import shutil
import signal
import time
from pathlib import Path


_ACTIVE_SOLO12_RACE_SOURCE_BACKUPS: list[tuple[Path, Path]] = []
_ACTIVE_SOLO12_RACE_SOURCE_RESTORED = True

def _jsonify(obj, max_str=4000):
    """Convert nested config objects to something wandb can store in config (JSON-safe)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, str) and len(obj) > max_str:
            return obj[:max_str] + "…(truncated)"
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x, max_str=max_str) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # W&B config keys must be strings
            ks = str(k)
            out[ks] = _jsonify(v, max_str=max_str)
        return out

    # isaaclab @configclass often supports to_dict()
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _jsonify(obj.to_dict(), max_str=max_str)
        except Exception:
            pass

    # fallback: best-effort repr
    s = repr(obj)
    if len(s) > max_str:
        s = s[:max_str] + "…(truncated)"
    return s


def _wandb_snapshot(
    log_dir: str,
    env_cfg,
    agent_cfg,
    args_cli,
    env_cfg_py: str,
    env_py: str,
    extra_files: list[str] | None = None,
):
    """Attach config + code to the active W&B run."""
    import wandb

    run = wandb.run
    if run is None:
        print("[WARN] wandb.run is None (run not initialized yet). Snapshot will remain only on disk.")
        return

    # 1) Put env_cfg into W&B Config tab (JSON-safe)
    env_cfg_dict = _jsonify(env_cfg)
    # Also useful: store CLI args + agent_cfg (sanitized)
    cfg_payload = {
        "cli": _jsonify(vars(args_cli)),
        "env_cfg": env_cfg_dict,
        "agent_cfg": _jsonify(agent_cfg),
    }
    run.config.update(cfg_payload, allow_val_change=True)

    # 2) Upload files (env/agent yaml + python sources) as an artifact
    art = wandb.Artifact(name="borinot-snapshot", type="repro")
    params_dir = Path(log_dir) / "params"
    if params_dir.exists():
        art.add_dir(str(params_dir))  # env.yaml, agent.yaml, etc.

    files_to_snapshot: list[str] = []
    for file_path in [env_cfg_py, env_py, *(extra_files or [])]:
        if file_path and Path(file_path).exists() and file_path not in files_to_snapshot:
            files_to_snapshot.append(file_path)
            art.add_file(file_path)

    # (optional) also add a copy under log_dir for local reproducibility
    snap_dir = Path(log_dir) / "code_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files_to_snapshot:
        shutil.copy2(file_path, snap_dir / Path(file_path).name)

    run.log_artifact(art)

    # Alternative (instead of artifact): show up under the run "Files" panel
    # wandb.save(str(params_dir / "env.yaml"), base_path=log_dir)
    # wandb.save(str(params_dir / "agent.yaml"), base_path=log_dir)
    # wandb.save(env_cfg_py, base_path=log_dir)
    # wandb.save(env_py, base_path=log_dir)


def _infer_wandb_run_id_from_checkpoint(checkpoint_path: str) -> str | None:
    """Infer a W&B run id from checkpoint names like ``0425_anjyauoc_best_model.pt``."""
    stem = Path(checkpoint_path).stem.lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", stem) if token]
    skip_tokens = {"best", "agent", "model", "lastagent", "last", "overnight", "checkpoint", "ckpt"}
    for token in tokens:
        if token in skip_tokens or token.isdigit():
            continue
        if 7 <= len(token) <= 8 and any(ch.isalpha() for ch in token):
            return token
    return None


def _is_downloaded_cluster_checkpoint(checkpoint_path: str, isaaclab_root: Path) -> bool:
    """Return True for flat downloaded checkpoints stored in ``logs/skrl/checkpoints``."""
    try:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        cluster_checkpoint_dir = (isaaclab_root / "logs" / "skrl" / "checkpoints").resolve()
    except OSError:
        return False
    return checkpoint.parent == cluster_checkpoint_dir


def _download_training_snapshot_from_wandb(
    *,
    entity: str,
    project: str,
    run_id: str,
    target_dir: Path,
    filenames: tuple[str, ...],
    max_attempts: int = 3,
) -> dict[str, Path]:
    """Download the training code snapshot artifact and cache selected files locally."""
    target_dir.mkdir(parents=True, exist_ok=True)
    cached = {name: target_dir / name for name in filenames}
    if all(path.exists() for path in cached.values()):
        return cached

    import wandb

    run_path = f"{entity}/{project}/{run_id}"
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            api = wandb.Api(timeout=30)
            run = api.run(run_path)
            artifacts = list(run.logged_artifacts())
            artifact = next(
                (
                    art
                    for art in artifacts
                    if getattr(art, "type", None) == "repro" and art.name.split(":", 1)[0] == "borinot-snapshot"
                ),
                None,
            )
            if artifact is None:
                artifact_names = ", ".join(f"{art.name} ({getattr(art, 'type', '?')})" for art in artifacts)
                raise RuntimeError(
                    f"W&B run '{run_path}' has no 'borinot-snapshot' repro artifact. "
                    f"Logged artifacts: {artifact_names or 'none'}"
                )

            artifact_dir = Path(artifact.download(root=str(target_dir / "_artifact")))
            found: dict[str, Path] = {}
            for name in filenames:
                matches = list(artifact_dir.rglob(name))
                if not matches:
                    raise RuntimeError(f"Artifact '{artifact.name}' does not contain '{name}'.")
                cached_path = target_dir / name
                shutil.copy2(matches[0], cached_path)
                found[name] = cached_path
            return found
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                sleep_s = float(attempt)
                print(
                    f"[WARN] Failed to fetch W&B snapshot for '{run_path}' on attempt "
                    f"{attempt}/{max_attempts}: {exc}. Retrying in {sleep_s:.1f}s...",
                    flush=True,
                )
                time.sleep(sleep_s)

    raise RuntimeError(f"Failed to fetch W&B snapshot for '{run_path}': {last_error}") from last_error


def _copy_with_bak_restore(source_by_name: dict[str, Path], target_dir: Path) -> list[tuple[Path, Path]]:
    """Copy source files into target_dir after creating ``.bak`` backups."""
    backups: list[tuple[Path, Path]] = []
    try:
        for name, source in source_by_name.items():
            target = target_dir / name
            backup = target.with_suffix(target.suffix + ".bak")
            if backup.exists():
                raise RuntimeError(
                    f"Refusing to overwrite existing backup '{backup}'. Restore/remove it before retrying."
                )
            if not target.exists():
                raise FileNotFoundError(f"Expected target source file does not exist: {target}")
            shutil.copy2(target, backup)
            backups.append((target, backup))
            shutil.copy2(source, target)
    except Exception:
        _restore_bak_files(backups)
        raise
    return backups


def _restore_bak_files(backups: list[tuple[Path, Path]]) -> None:
    """Restore files from ``.bak`` backups and remove the backups."""
    for target, backup in reversed(backups):
        if backup.exists():
            shutil.copy2(backup, target)
            backup.unlink()


def solo12_race_sources_need_restore() -> bool:
    """Return True when a training snapshot is active and local source backups still exist."""
    return (
        bool(_ACTIVE_SOLO12_RACE_SOURCE_BACKUPS)
        and not _ACTIVE_SOLO12_RACE_SOURCE_RESTORED
        and any(backup.exists() for _, backup in _ACTIVE_SOLO12_RACE_SOURCE_BACKUPS)
    )


def restore_solo12_race_sources() -> bool:
    """Restore local Solo12 race source files from active ``.bak`` backups."""
    global _ACTIVE_SOLO12_RACE_SOURCE_BACKUPS, _ACTIVE_SOLO12_RACE_SOURCE_RESTORED

    if not solo12_race_sources_need_restore():
        return False
    _restore_bak_files(_ACTIVE_SOLO12_RACE_SOURCE_BACKUPS)
    _ACTIVE_SOLO12_RACE_SOURCE_RESTORED = True
    print("[INFO] Restored local Solo12 race source files from .bak backups.", flush=True)
    return True


def _install_restore_handlers(backups: list[tuple[Path, Path]]) -> None:
    """Install fallback source-file restore at normal exit or Ctrl+C/SIGTERM."""
    global _ACTIVE_SOLO12_RACE_SOURCE_BACKUPS, _ACTIVE_SOLO12_RACE_SOURCE_RESTORED

    _ACTIVE_SOLO12_RACE_SOURCE_BACKUPS = list(backups)
    _ACTIVE_SOLO12_RACE_SOURCE_RESTORED = False

    atexit.register(restore_solo12_race_sources)

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handler = signal.getsignal(sig)

        def handler(signum, frame, previous_handler=previous_handler):  # noqa: ARG001
            restore_solo12_race_sources()
            if callable(previous_handler):
                previous_handler(signum, frame)
            raise SystemExit(128 + signum)

        signal.signal(sig, handler)


def sync_solo12_race_sources_from_wandb(
    *,
    checkpoint_path: str | None,
    task: str | None,
    entity: str,
    project: str,
    run_id: str | None = None,
    force: bool = False,
) -> str | None:
    """
    For downloaded cluster checkpoints, temporarily swap Solo12 race source files with
    the exact files captured in the training run's W&B snapshot artifact.

    Returns the inferred/used run id when a swap was installed; otherwise ``None``.
    """
    if not checkpoint_path:
        return None

    isaaclab_root = Path.cwd().resolve()
    if not force and not _is_downloaded_cluster_checkpoint(checkpoint_path, isaaclab_root):
        return None

    if not task or "Solo12-Race" not in task:
        return None

    run_id = run_id or _infer_wandb_run_id_from_checkpoint(checkpoint_path)
    if run_id is None:
        raise RuntimeError(
            f"Could not infer W&B run id from checkpoint '{checkpoint_path}'. "
            "Pass --training_wandb_run_id explicitly."
        )

    filenames = ("solo12_race_env_cfg.py", "solo12_race_env.py")
    cache_dir = isaaclab_root / "logs" / "skrl" / "files" / run_id
    source_by_name = _download_training_snapshot_from_wandb(
        entity=entity,
        project=project,
        run_id=run_id,
        target_dir=cache_dir,
        filenames=filenames,
    )
    race_source_dir = isaaclab_root / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "direct" / "solo12_race"
    backups = _copy_with_bak_restore(source_by_name, race_source_dir)
    _install_restore_handlers(backups)
    print(
        f"[INFO] Using Solo12 race source snapshot from W&B run '{entity}/{project}/{run_id}' "
        f"for this play session. Cached in: {cache_dir}",
        flush=True,
    )
    return run_id
