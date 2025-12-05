from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from explainibility_agents.pipeline import top_k_for_date_from_log
from explainibility_agents.tradingagents.combined_weight_agent import WeightSynthesisAgent

from .config import XAIRequest
from .registry import register_mcp_tool


def run_trading_job(cfg: XAIRequest) -> Dict[str, object]:
    holdings = top_k_for_date_from_log(
        Path(cfg.monthly_log_csv),
        cfg.date,
        top_k=cfg.top_k,
        run_id=cfg.monthly_run_id,
    )
    agent = WeightSynthesisAgent()
    reports: List[Dict[str, object]] = []

    for row in holdings:
        ticker = str(row.get("ticker", "?")).strip().upper()
        weight = float(row.get("weight", 0.0))
        as_of = row.get("as_of")
        try:
            report = agent.generate_report(
                ticker,
                weight,
                as_of=as_of,
                lookback_days=cfg.lookback_days,
                max_articles=8,
                use_llm=cfg.llm,
                llm_model=cfg.llm_model,
            )
            md_path = cfg.output_dir / f"{ticker}_summary_mcp.md"
            md_path.write_text(
                report.to_markdown(include_components=True, include_metrics=True, include_articles=True),
                encoding="utf-8",
            )
            reports.append(
                {
                    "ticker": ticker,
                    "success": True,
                    "weight": weight,
                    "as_of": as_of,
                    "output_path": str(md_path),
                    "summary_points": list(report.summary_points),
                    "llm_used": bool(report.generated_via_llm),
                }
            )
        except Exception as exc:  # noqa: BLE001
            reports.append(
                {
                    "ticker": ticker,
                    "success": False,
                    "weight": weight,
                    "as_of": as_of,
                    "error": str(exc),
                }
            )

    return {"holdings": holdings, "reports": reports}


RUN_TRADING_SCHEMA = {
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
    name="run_trading_agent",
    description="Run the WeightSynthesisAgent for each focus holding and persist markdown reports.",
    schema=RUN_TRADING_SCHEMA,
)
def mcp_run_trading_agent(payload: Dict[str, object]) -> Dict[str, object]:
    cfg = XAIRequest.from_payload(payload)
    return run_trading_job(cfg)
