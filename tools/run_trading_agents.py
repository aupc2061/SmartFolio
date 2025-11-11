#!/usr/bin/env python
"""
SmartFolio TradingAgents Runner
=================================

This orchestrator executes the Fundamental, News, and Combined agents
to produce explainability summaries for a portfolio allocation CSV.

Usage:
  python tools/run_trading_agents.py \
      --allocation-csv allocation.csv \
      --include-components \
      --print-summaries \
      --llm \
      --llm-model gemini-2.0-flash
"""

from __future__ import annotations
import argparse
import pandas as pd
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.markdown import Markdown

# ---------------------------------------------------------------------
# 🔧 Fix PYTHONPATH so the script finds trading_agent/tradingagents/
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # /SmartFolio
AGENT_ROOT = ROOT / "trading_agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

# ---------------------------------------------------------------------
# ✅ Import TradingAgents
# ---------------------------------------------------------------------
from tradingagents.combined_weight_agent import WeightSynthesisAgent
from tradingagents.fundamental_agent import FundamentalWeightAgent
from tradingagents.news_agent import NewsWeightReviewAgent

console = Console()


# ---------------------------------------------------------------------
# 🔹 CLI Argument Parser
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run SmartFolio TradingAgents suite.")
    p.add_argument(
        "--allocation-csv",
        default="allocation.csv",
        help="CSV file with at least columns: ticker,weight (default: allocation.csv)",
    )
    p.add_argument(
        "--trading-agent-root",
        default=str(AGENT_ROOT),
        help="Path to trading_agent package (default: trading_agent)",
    )
    p.add_argument(
        "--include-components",
        action="store_true",
        help="Include detailed fundamentals and news components in markdown output.",
    )
    p.add_argument(
        "--include-metrics",
        action="store_true",
        help="Include the fundamental metrics table in the report.",
    )
    p.add_argument(
        "--include-articles",
        action="store_true",
        help="Include news headline tables in the report.",
    )
    p.add_argument(
        "--print-summaries",
        action="store_true",
        help="Print markdown reports directly to console.",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM synthesis via Gemini 2.0 Flash for the unified summary.",
    )
    p.add_argument(
        "--llm-model",
        default="gemini-2.0-flash",
        help="Gemini model name (default: gemini-2.0-flash).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel threads to run agent computations.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# 🧩 Worker Function for Each Ticker
# ---------------------------------------------------------------------
def run_agent_for_row(row, args):
    ticker = str(row["ticker"]).strip().upper()
    weight = float(row["weight"])
    try:
        agent = WeightSynthesisAgent()
        report = agent.generate_report(
            ticker,
            weight,
            as_of=None,
            lookback_days=7,
            max_articles=8,
            use_llm=args.llm,
            llm_model=args.llm_model,
        )

        markdown_text = report.to_markdown(
            include_components=args.include_components,
            include_metrics=args.include_metrics,
            include_articles=args.include_articles,
        )

        # Save to explainability_results/
        out_dir = Path("explainability_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ticker}_summary.md"
        out_path.write_text(markdown_text, encoding="utf-8")

        return ticker, True, report.generated_via_llm, out_path, markdown_text
    except Exception as e:
        return ticker, False, False, None, f"[ERROR] {ticker}: {e}"


# ---------------------------------------------------------------------
# 🚀 Main Orchestration Function
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    requested_root = Path(args.trading_agent_root).expanduser().resolve()
    if requested_root != AGENT_ROOT.resolve() and str(requested_root) not in sys.path:
        sys.path.insert(0, str(requested_root))

    csv_path = Path(args.allocation_csv)
    if not csv_path.exists():
        console.print(f"[red]❌ Missing allocation CSV: {csv_path}[/red]")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if "ticker" not in df.columns or "weight" not in df.columns:
        console.print("[red]❌ CSV must include columns 'ticker' and 'weight'[/red]")
        sys.exit(1)

    console.print(
        f"[bold cyan]🚀 Running TradingAgents for {len(df)} tickers...[/bold cyan]\n"
    )

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_agent_for_row, row, args): row for _, row in df.iterrows()}
        for fut in as_completed(futures):
            results.append(fut.result())

    # Summary
    success_count = sum(1 for _, ok, *_ in results if ok)
    fail_count = len(results) - success_count
    console.print(f"\n✅ [green]{success_count} succeeded[/green], ❌ [red]{fail_count} failed[/red].\n")

    # Console printing
    if args.print_summaries:
        for ticker, ok, via_llm, path, output in results:
            console.print(f"\n[bold yellow]{ticker}[/bold yellow]")
            if not ok:
                console.print(f"[red]{output}[/red]\n")
                continue
            console.print(Markdown(output))
            console.print(f"[dim]Saved: {path}[/dim]")
            if via_llm:
                console.print("[dim green]Generated via LLM (Gemini 2.0 Flash)[/dim]\n")

    # Write summary index CSV
    summary_data = [
        {
            "ticker": t,
            "success": ok,
            "llm_used": via_llm,
            "output_path": str(p) if p else "",
        }
        for t, ok, via_llm, p, _ in results
    ]
    summary_df = pd.DataFrame(summary_data)
    out_dir = Path("explainability_results")
    out_dir.mkdir(exist_ok=True)
    summary_csv = out_dir / "summary_index.csv"
    summary_df.to_csv(summary_csv, index=False)
    console.print(f"\n📊 [bold green]Summary index written to {summary_csv}[/bold green]\n")


# ---------------------------------------------------------------------
# 🔘 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
