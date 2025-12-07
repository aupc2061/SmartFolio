"""
Monthly automation agent:
- Checks manifest/state to detect a new full month.
- Runs monthly dataset update (update_monthly_dataset.py) to build daily pickles/corr/manifest.
- Rebuilds monthly shards via Pathway monthly builder.
- Triggers fine-tune on the latest unprocessed shard.

Designed to be called from cron/systemd. Supports dry-run and locking to avoid overlap.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

STATE_FILE = "monthly_agent_state.json"
LOCK_FILE = "monthly_agent.lock"
DEFAULT_MANIFEST = "monthly_manifest.json"


def _default_manifest_path(market: str, horizon: int | str, relation_type: str) -> Path:
    return Path("dataset_default") / f"data_train_predict_{market}" / f"{horizon}_{relation_type}" / DEFAULT_MANIFEST


def _resolve_manifest_path(manifest_arg: str, market: str, horizon: int | str, relation_type: str) -> Path:
    """
    Prefer an explicitly provided manifest if it exists; otherwise use the canonical dataset path.
    If a relative filename is provided but does not exist, place it under the dataset dir.
    """
    if manifest_arg:
        candidate = Path(manifest_arg).expanduser()
        if candidate.is_file():
            return candidate
        if not candidate.is_absolute():
            alt = _default_manifest_path(market, horizon, relation_type).with_name(candidate.name)
            if alt.is_file():
                return alt
    return _default_manifest_path(market, horizon, relation_type)


def _load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def _save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def _acquire_lock() -> bool:
    if os.path.exists(LOCK_FILE):
        try:
            mtime = os.path.getmtime(LOCK_FILE)
            if time.time() - mtime > 24 * 3600:
                os.remove(LOCK_FILE)
            else:
                return False
        except Exception:
            return False
    try:
        with open(LOCK_FILE, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
        return True
    except Exception:
        return False


def _release_lock() -> None:
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except Exception:
        pass


def _read_manifest(manifest_path: str) -> Dict[str, Any]:
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _latest_month_from_manifest(manifest: Dict[str, Any]) -> Optional[str]:
    shards = manifest.get("monthly_shards", {})
    if isinstance(shards, dict):
        months = list(shards.keys())
    else:
        months = [s.get("month") for s in shards if s.get("month")]
    months = [m for m in months if m]
    return max(months) if months else None


def _run_cmd(cmd: str, dry_run: bool) -> int:
    print(f"[agent] Running: {cmd}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, shell=True)
    return proc.returncode


def _detect_new_month(manifest: Dict[str, Any]) -> Optional[str]:
    latest = _latest_month_from_manifest(manifest)
    if not latest:
        return None
    # Parse YYYY-MM and compute if the next month has started
    try:
        dt = datetime.datetime.strptime(latest + "-01", "%Y-%m-%d")
        next_month = (dt + datetime.timedelta(days=32)).replace(day=1)
        today = datetime.datetime.utcnow()
        if today.year > next_month.year or (today.year == next_month.year and today.month >= next_month.month):
            return next_month.strftime("%Y-%m")
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Monthly automation agent")
    parser.add_argument("--market", required=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--relation-type", default="hy")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Path to monthly_manifest.json")
    parser.add_argument("--data-build-flags", default="--no-pathway-rolling --no-pathway-macro", help="Flags for update_monthly_dataset.py")
    parser.add_argument("--fine-tune-flags", default="--policy HGAT --discover_months_with_pathway --irl_epochs 1 --rl_timesteps 1 --fine_tune_steps 1", help="Flags for main.py fine-tune")
    parser.add_argument("--dry-run", action="store_true", help="Log what would run, but do nothing")
    args = parser.parse_args()

    manifest_path = _resolve_manifest_path(args.manifest, args.market, args.horizon, args.relation_type)

    if not _acquire_lock():
        print("[agent] Lock exists, aborting.")
        return

    try:
        state = _load_state()
        manifest = _read_manifest(str(manifest_path))
        last_processed = state.get("last_processed_month") or manifest.get("last_fine_tuned_month")
        latest_manifest_month = _latest_month_from_manifest(manifest)

        # Detect whether a new month is available by calendar or manifest gap.
        new_month = _detect_new_month(manifest)
        needs_run = False
        if new_month:
            needs_run = True
        elif latest_manifest_month and last_processed and latest_manifest_month > last_processed:
            needs_run = True
        elif latest_manifest_month and not last_processed:
            needs_run = True
        else:
            # no manifest? trigger a run
            if not os.path.exists(manifest_path):
                needs_run = True

        if not needs_run:
            print("[agent] No new month detected; exiting.")
            return

        data_cmd = (
            f"python -m gen_data.update_monthly_dataset "
            f"--market {args.market} "
            f"--horizon {args.horizon} "
            f"--relation-type {args.relation_type} "
            f"{args.data_build_flags}"
        )
        rc = _run_cmd(data_cmd, args.dry_run)
        if rc != 0:
            print(f"[agent] Data build failed with code {rc}")
            return

        ft_cmd = (
            f"python main.py --run_monthly_fine_tune "
            f"--market {args.market} "
            f"--horizon {args.horizon} "
            f"--relation_type {args.relation_type} "
            f"{args.fine_tune_flags} "
            f"--manifest {manifest_path}"
        )
        rc = _run_cmd(ft_cmd, args.dry_run)
        if rc != 0:
            print(f"[agent] Fine-tune run failed with code {rc}")
            return
        manifest_after = _read_manifest(str(manifest_path))
        last_ft = manifest_after.get("last_fine_tuned_month") or _latest_month_from_manifest(manifest_after)
        state["last_processed_month"] = last_ft
        _save_state(state)
        print(f"[agent] Completed. Last processed month: {last_ft}")

    finally:
        _release_lock()

if __name__ == "__main__":
    main()