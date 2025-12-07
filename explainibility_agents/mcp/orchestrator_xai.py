from __future__ import annotations

import json
from pathlib import Path
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

    # Read markdown and JSON content instead of storing paths
    final_md_path = cfg.output_dir / "explainability_final_results.md"
    final_md_content = None
    if final_md_path.exists():
        final_md_content = final_md_path.read_text(encoding="utf-8")

    final_json_path = cfg.output_dir / "explainability_final_results.json"
    final_json_content = None
    if final_json_path.exists():
        try:
            final_json_content = json.loads(final_json_path.read_text(encoding="utf-8"))
        except:
            final_json_content = None

    # Read individual stock markdown files
    stock_reports = []
    if index_payload and "trading_agents" in index_payload:
        for trading_agent in index_payload.get("trading_agents", []):
            ticker = trading_agent.get("ticker")
            output_path = trading_agent.get("output_path")
            if ticker and output_path:
                # Try to read from the output_path (which might be a full path or relative)
                stock_md_path = Path(output_path)
                if not stock_md_path.exists():
                    # Try relative to output_dir
                    stock_md_path = cfg.output_dir / f"{ticker}_summary.md"
                
                if stock_md_path.exists():
                    try:
                        stock_md_content = stock_md_path.read_text(encoding="utf-8")
                        stock_reports.append({
                            "ticker": ticker,
                            "markdown": stock_md_content,
                            "weight": trading_agent.get("weight"),
                            "as_of": trading_agent.get("as_of"),
                            "summary_points": trading_agent.get("summary_points", []),
                            "llm_used": trading_agent.get("llm_used", False),
                            "success": trading_agent.get("success", False)
                        })
                    except Exception as e:
                        # If file read fails, still store the report without markdown
                        stock_reports.append({
                            "ticker": ticker,
                            "markdown": None,
                            "weight": trading_agent.get("weight"),
                            "as_of": trading_agent.get("as_of"),
                            "summary_points": trading_agent.get("summary_points", []),
                            "llm_used": trading_agent.get("llm_used", False),
                            "success": trading_agent.get("success", False),
                            "error": f"Could not read markdown: {str(e)}"
                        })

    return {
        "index": index_payload,
        "final_markdown": final_md_content,
        "final_json": final_json_content,
        "stock_reports": stock_reports,
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
