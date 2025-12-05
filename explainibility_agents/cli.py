"""CLI entry-point for the LangGraph explainibility pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from explainibility_agents.config import ExplainabilityConfig
from explainibility_agents.pipeline import run_explainibility_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LangChain-powered explainibility orchestrator.")
    parser.add_argument("--date", required=True, help="Snapshot date to analyze (YYYY-MM-DD)")
    parser.add_argument("--monthly-log-csv", required=True, help="Path to monthly final weights CSV")
    parser.add_argument("--monthly-run-id", default=None, help="Optional run_id filter when CSV tracks multiple runs")
    parser.add_argument("--model-path", required=True, help="Path to trained PPO checkpoint")
    parser.add_argument("--market", default="custom")
    parser.add_argument("--data-root", default="dataset_default")
    parser.add_argument("--top-k", type=int, default=5, help="Number of holdings to explain (default 5)")
    parser.add_argument("--lookback-days", type=int, default=30, help="Days before --date for explainability datasets")
    parser.add_argument("--llm", action="store_true", help="Enable LLM generation for narratives")
    parser.add_argument("--latent", action="store_true", help="Enable latent factor analysis")
    parser.add_argument("--llm-model", default="gpt-4.1-mini", help="Chat model to use via LangChain")
    parser.add_argument("--output-dir", default="explainability_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExplainabilityConfig(
        date=args.date,
        monthly_log_csv=Path(args.monthly_log_csv).expanduser(),
        model_path=args.model_path,
        market=args.market,
        data_root=args.data_root,
        top_k=int(args.top_k or 5),
        lookback_days=int(args.lookback_days or 60),
        llm=bool(args.llm),
        llm_model=args.llm_model,
        output_dir=Path(args.output_dir),
        monthly_run_id=args.monthly_run_id,
        latent=bool(args.latent),
    )
    run_explainibility_pipeline(cfg)


if __name__ == "__main__":
    main()
