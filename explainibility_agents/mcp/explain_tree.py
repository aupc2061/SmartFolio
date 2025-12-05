from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from explainibility_agents import explain_tree as base_tree

from .config import XAIRequest
from .registry import register_mcp_tool


def _build_args(cfg: XAIRequest, start_date: str, end_date: str) -> List[str]:
    argv: List[str] = [
        "--model-path",
        cfg.model_path,
        "--market",
        cfg.market,
        "--test-start-date",
        start_date,
        "--test-end-date",
        end_date,
        "--data-root",
        cfg.data_root,
        "--device",
        "cpu",
        "--save-joblib",
        "--save-summary",
        "--output-dir",
        str(cfg.output_dir),
        "--focus-log-csv",
        cfg.monthly_log_csv,
        "--focus-date",
        cfg.date,
        "--tickers-csv",
        "tickers.csv",
    ]
    argv.extend(["--max-steps", str(cfg.lookback_days)])
    if cfg.monthly_run_id:
        argv.extend(["--focus-run-id", cfg.monthly_run_id])
    if cfg.top_k:
        argv.extend(["--top-k-stocks", str(cfg.top_k)])

    augment_corr = cfg.extra.get("augment_corr")
    augment_ts_stats = cfg.extra.get("augment_ts_stats")

    if augment_corr is False:
        argv.append("--no-augment-corr")
    elif augment_corr is True:
        argv.append("--augment-corr")

    if augment_ts_stats is False:
        argv.append("--no-augment-ts-stats")
    elif augment_ts_stats is True:
        argv.append("--augment-ts-stats")

    return argv


def run_tree_job(cfg: XAIRequest) -> Dict[str, object]:
    start_date, end_date = cfg.extra.get("window", cfg.rolling_window())
    argv = _build_args(cfg, start_date, end_date)
    base_tree.main(argv)

    joblib_path = cfg.output_dir / f"explain_tree_{cfg.market}.joblib"
    summary_path = cfg.output_dir / f"explain_tree_{cfg.market}.json"
    summary_payload = None
    if summary_path.exists():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    return {
        "start_date": start_date,
        "end_date": end_date,
        "joblib_path": str(joblib_path) if joblib_path.exists() else None,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "summary": summary_payload,
    }


TREE_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "monthly_log_csv": {"type": "string"},
        "model_path": {"type": "string"},
        "market": {"type": "string"},
        "data_root": {"type": "string"},
        "top_k": {"type": "integer"},
        "lookback_days": {"type": "integer"},
        "llm": {"type": "boolean"},
        "llm_model": {"type": "string"},
        "output_dir": {"type": "string"},
        "monthly_run_id": {"type": "string"},
    },
    "required": ["date", "monthly_log_csv", "model_path", "lookback_days", "top_k", "market", "data_root", "output_dir", "monthly_run_id", "llm", "llm_model"],
}


@register_mcp_tool(
    name="generate_tree_surrogate",
    description="Fit and persist decision-tree surrogates for the focus holdings.",
    schema=TREE_SCHEMA,
)
def mcp_generate_tree_surrogate(payload: Dict[str, object]) -> Dict[str, object]:
    cfg = XAIRequest.from_payload(payload)
    return run_tree_job(cfg)
