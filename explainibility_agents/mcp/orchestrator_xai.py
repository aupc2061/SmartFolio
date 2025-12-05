from __future__ import annotations

import json
from typing import Dict

from explainibility_agents import orchestrator_xai as base_orchestrator

from .config import XAIRequest
from .registry import register_mcp_tool


def run_orchestrator_job(cfg: XAIRequest) -> Dict[str, object]:
    base_orchestrator.run_orchestrator(cfg.to_orchestrator_config())

    index_path = cfg.output_dir / "orchestrator_index.json"
    index_payload = None
    if index_path.exists():
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))

    final_md = cfg.output_dir / "explainability_final_results.md"

    return {
        "index_path": str(index_path) if index_path.exists() else None,
        "final_markdown": str(final_md) if final_md.exists() else None,
        "index": index_payload,
    }


RUN_XAI_SCHEMA = {
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
        "latent":{"type": "boolean"},
    },
    "required": ["date", "monthly_log_csv", "model_path", "lookback_days", "top_k", "market", "data_root", "output_dir", "monthly_run_id", "llm", "llm_model"],
}


@register_mcp_tool(
    name="run_xai_orchestrator",
    description="Run the full SmartFolio explainability pipeline and return artifact pointers.",
    schema=RUN_XAI_SCHEMA,
)
def mcp_run_xai_orchestrator(payload: Dict[str, object]) -> Dict[str, object]:
    cfg = XAIRequest.from_payload(payload)
    return run_orchestrator_job(cfg)
