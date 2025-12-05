# pyright: reportGeneralTypeIssues=false, reportMissingImports=false

"""LangChain + LangGraph explainibility pipeline."""

from __future__ import annotations

import datetime as _dt
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import joblib  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from langchain_core.prompts import ChatPromptTemplate  # type: ignore[import]
from langgraph.graph import END, StateGraph  # type: ignore[import]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explainibility_agents.config import ExplainabilityConfig, Holding
from explainibility_agents import explain_tree as explain_tree_mod
from explainibility_agents import explain_latent as explain_latent_mod
from explainibility_agents.tradingagents.combined_weight_agent import WeightSynthesisAgent

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from langchain_openai import ChatOpenAI  # type: ignore[import]
else:
    ChatOpenAI = Any  # fallback to keep runtime import optional

TREE_SYSTEM_PROMPT = """
You are a Financial Analyst writing a simplified newsletter for retail investors. Your goal is to explain an AI's trading strategy in plain English, removing all technical jargon.

Follow these strict rules:
1. No math terms such as threshold, node, coefficient, z-score.
2. Do not quote raw numbers; describe them qualitatively ("sharp spike", "steady dip").
3. Ignore any average weight information.
4. Keep every sentence short and direct.

Return per ticker sections that look like this:

**ICICIBANK.NS**
* **The Strategy**: Sector rotation using peer strength
* **The Logic**: The model likes this bank when SBI's volume heats up, treating that as confirmation for the whole sector. It backs off when Infosys is sliding because that signals broader tech-led fear.
"""

FINAL_SYSTEM_PROMPT = (
    "You are a senior portfolio strategist. Combine fundamental takeaways, tree-based rules, and latent factor flags "
    "into one concise note per stock. Use at most three short bullet points (under 120 total words) per ticker and never offer services, monitoring, or extra actions."
)


PipelineState = Dict[str, Any]


TREE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", TREE_SYSTEM_PROMPT.strip()),
        ("human", "Summarize the following decision tree payload in the requested format.\n{payload}"),
    ]
)

FINAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", FINAL_SYSTEM_PROMPT),
        (
            "human",
            "Ticker: {ticker}\nWeight: {weight}\nAs Of: {as_of}\n"
            "Trading Agent Markdown:\n{trading_markdown}\n\n"
            "Tree Signal Summary:\n{tree_rules}\n\nLatent Notes:\n{latent_notes}\n",
        ),
    ]
)


def rolling_window(date_str: str, lookback_days: int) -> Tuple[str, str]:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    end_date = _dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    start_date = end_date - _dt.timedelta(days=lookback_days - 1)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def top_k_for_date_from_log(
    monthly_log_csv: Path,
    target_date: str,
    *,
    top_k: int = 5,
    run_id: str | None = None,
) -> List[dict]:
    df = pd.read_csv(monthly_log_csv)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if run_id and "run_id" in df.columns:
        df = df[df["run_id"] == run_id]
    if df.empty:
        raise RuntimeError(f"No rows remain after applying run_id filter for {monthly_log_csv}")

    date_cols = [col for col in ("as_of", "date") if col in df.columns]
    if not date_cols:
        raise RuntimeError("monthly log CSV must include either 'as_of' or 'date' column")

    target = pd.to_datetime(target_date).date()
    for col in date_cols:
        series = pd.to_datetime(df[col], errors="coerce")
        mask = series.dt.date == target
        if mask.any():
            filtered = df[mask].copy()
            filtered["as_of"] = series[mask].dt.strftime("%Y-%m-%d")
            filtered = filtered.sort_values("weight", ascending=False).head(top_k)
            return filtered.to_dict(orient="records")

    raise RuntimeError(f"No holdings found for target date {target_date} in {monthly_log_csv}")


def collect_holdings(cfg: ExplainabilityConfig) -> List[Holding]:
    rows = top_k_for_date_from_log(
        cfg.monthly_log_csv,
        cfg.date,
        top_k=cfg.top_k,
        run_id=cfg.monthly_run_id,
    )
    holdings = [
        Holding(
            ticker=str(row["ticker"]).strip().upper(),
            weight=float(row["weight"]),
            as_of=row.get("as_of"),
        )
        for row in rows
    ]
    return holdings


def run_explain_tree_module(
    cfg: ExplainabilityConfig,
    start_date: str,
    end_date: str,
    out_dir: Path,
) -> Path:
    argv = [
        "--model-path",
        str(cfg.model_path),
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
        str(out_dir),
        "--top-k-stocks",
        str(cfg.top_k),
        "--focus-log-csv",
        str(cfg.monthly_log_csv),
        "--focus-date",
        cfg.date,
        "--tickers-csv",
        "tickers.csv",
    ]
    argv.extend(["--max-steps", str(cfg.lookback_days)])
    if cfg.monthly_run_id:
        argv.extend(["--focus-run-id", cfg.monthly_run_id])
    try:
        explain_tree_mod.main(argv)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"explain_tree failed: {exc}") from exc
    return out_dir / f"explain_tree_{cfg.market}.joblib"


def generate_tree_narrative(
    snapshot: Path,
    out_dir: Path,
    llm: Optional[ChatOpenAI],
) -> Tuple[Optional[Path], Optional[str]]:
    if not snapshot.exists():
        return None, None

    payload = joblib.load(snapshot)
    if not isinstance(payload, dict):
        raise RuntimeError("Tree snapshot must contain a dict")

    prompt_payload = json.dumps(payload, indent=2, default=str)
    prompt_path = out_dir / "tree_prompt_payload.json"
    prompt_path.write_text(prompt_payload, encoding="utf-8")

    if llm is None:
        return None, "Tree narrative requires LLM access."

    messages = TREE_PROMPT.format_prompt(payload=prompt_payload).to_messages()
    response = llm.invoke(messages)
    tree_markdown = response.content if hasattr(response, "content") else str(response)
    narrative_path = out_dir / "explainability_tree_narrative.md"
    narrative_path.write_text(tree_markdown, encoding="utf-8")
    return narrative_path, tree_markdown


def run_latent_module(
    cfg: ExplainabilityConfig,
    start_date: str,
    end_date: str,
    out_dir: Path,
) -> Optional[Dict[str, object]]:
    if not cfg.latent:
        return None
    latent_dir = out_dir / "latent_factors"
    try:
        result = explain_latent_mod.run_latent_pipeline(
            model_path=cfg.model_path,
            market=cfg.market,
            start_date=start_date,
            end_date=end_date,
            data_root=cfg.data_root,
            output_dir=latent_dir,
            device="cpu",
            llm=cfg.llm,
            llm_model=cfg.llm_model,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}


def run_trading_agents(
    cfg: ExplainabilityConfig,
    holdings: Iterable[Holding],
    out_dir: Path,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    agent = WeightSynthesisAgent()
    for holding in holdings:
        try:
            report = agent.generate_report(
                holding.ticker,
                holding.weight,
                as_of=holding.as_of,
                lookback_days=30,
                max_articles=8,
                use_llm=cfg.llm,
                llm_model=cfg.llm_model,
            )
            md = report.to_markdown(include_components=True, include_metrics=True, include_articles=True)
            out_path = out_dir / f"{holding.ticker}_summary.md"
            out_path.write_text(md, encoding="utf-8")
            rows.append(
                {
                    "ticker": holding.ticker,
                    "success": True,
                    "output_path": str(out_path),
                    "summary_points": list(report.summary_points),
                    "llm_used": bool(report.generated_via_llm),
                    "as_of": holding.as_of,
                    "weight": holding.weight,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "ticker": holding.ticker,
                    "success": False,
                    "error": str(exc),
                    "as_of": holding.as_of,
                    "weight": holding.weight,
                }
            )
    return rows


def _load_tree_per_stock(snapshot_path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if not snapshot_path or not snapshot_path.exists():
        return {}
    payload = joblib.load(snapshot_path)
    per_stock = payload.get("per_stock", {}) if isinstance(payload, dict) else {}
    mapping: Dict[str, Dict[str, object]] = {}
    for entry in per_stock.values():
        ticker = entry.get("ticker")
        if ticker:
            mapping[str(ticker).upper()] = entry
    return mapping


def _format_tree_rules(entry: Optional[Dict[str, object]]) -> str:
    if not entry:
        return "Tree surrogate signal unavailable."
    rules_obj = entry.get("rules")
    if isinstance(rules_obj, str) and rules_obj.strip():
        return rules_obj.strip()
    return json.dumps(entry, indent=2, default=str)


def _read_trading_markdown(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return "Trading agent markdown missing."


def build_final_reports(
    holdings: List[Holding],
    trading_rows: List[Dict[str, object]],
    tree_snapshot: Optional[Path],
    latent_result: Optional[Dict[str, object]],
    llm: Optional[ChatOpenAI],
) -> List[Dict[str, object]]:
    rows_by_ticker = {row.get("ticker"): row for row in trading_rows}
    tree_per_stock = _load_tree_per_stock(tree_snapshot)
    latent_summary = None
    if isinstance(latent_result, dict):
        latent_summary = latent_result.get("summary_md") or latent_result.get("summary")
    results: List[Dict[str, object]] = []

    for holding in holdings:
        trade = rows_by_ticker.get(holding.ticker)
        tree_entry = tree_per_stock.get(holding.ticker)
        latent_notes = str(latent_summary or "Latent factors disabled.")

        if llm and trade and trade.get("success"):
            prompt_value = FINAL_PROMPT.format_prompt(
                ticker=holding.ticker,
                weight=f"{holding.weight*100:.2f}%",
                as_of=holding.as_of or "n/a",
                trading_markdown=_read_trading_markdown(str(trade["output_path"])),
                tree_rules=_format_tree_rules(tree_entry),
                latent_notes=latent_notes,
            )
            response = llm.invoke(prompt_value.to_messages())
            content = response.content if hasattr(response, "content") else str(response)
            results.append({"ticker": holding.ticker, "success": True, "output": content, "llm_used": True})
            continue

        fallback_parts = [f"Summary for {holding.ticker}."]
        if trade and trade.get("success"):
            raw_points = trade.get("summary_points") or []
            points = [str(point) for point in raw_points if point]
            if points:
                fallback_parts.append("Key takeaways: " + "; ".join(points))
        if tree_entry:
            fallback_parts.append("Tree hint: " + _format_tree_rules(tree_entry).splitlines()[0])
        fallback_parts.append(latent_notes)
        results.append({"ticker": holding.ticker, "success": True, "output": " ".join(fallback_parts), "llm_used": False})
    return results


def write_final_markdown(
    cfg: ExplainabilityConfig,
    holdings: List[Holding],
    tree_text: Optional[str],
    latent_text: Optional[str],
    trading_rows: List[Dict[str, object]],
    final_reports: List[Dict[str, object]],
    destination: Path,
) -> None:
    trading_map = {row.get("ticker"): row for row in trading_rows}
    final_map = {row.get("ticker"): row for row in final_reports}

    lines: List[str] = []
    lines.append(f"# Explainability Recap — {cfg.market} ({cfg.date})")
    lines.append("\n## Top Holdings")
    for rank, holding in enumerate(holdings, start=1):
        lines.append(f"{rank}. **{holding.ticker}** — weight {holding.weight*100:.2f}% (as_of {holding.as_of or 'n/a'})")

    lines.append("\n## Tree Narrative")
    lines.append(tree_text or "(tree narrative unavailable)")

    if latent_text:
        lines.append("\n## Latent Factors")
        lines.append(latent_text)

    lines.append("\n## Per-Stock Insights")
    for holding in holdings:
        lines.append(f"\n### {holding.ticker}")
        trade = trading_map.get(holding.ticker)
        if trade and trade.get("success"):
            lines.append("**Trading Agent Highlights**")
            summary_points = trade.get("summary_points")
            if isinstance(summary_points, list):
                for point in summary_points:
                    lines.append(f"- {point}")
            lines.append(f"Report: {trade.get('output_path')}")
        elif trade:
            lines.append(f"Trading agent failed: {trade.get('error','unknown')}")
        else:
            lines.append("Trading agent output missing.")

        final = final_map.get(holding.ticker)
        if final and final.get("success"):
            tag = "LLM" if final.get("llm_used") else "Fallback"
            lines.append(f"**Final Synthesis ({tag})**\n{final.get('output')}")
        elif final:
            lines.append(f"Final synthesis failed: {final.get('error','unknown')}")
        else:
            lines.append("Final synthesis unavailable.")

    destination.write_text("\n".join(lines), encoding="utf-8")


def write_index_file(
    cfg: ExplainabilityConfig,
    holdings: List[Holding],
    trading_rows: List[Dict[str, object]],
    final_reports: List[Dict[str, object]],
    tree_snapshot: Optional[Path],
    tree_narrative: Optional[Path],
    latent_result: Optional[Dict[str, object]],
    final_markdown: Path,
    destination: Path,
) -> None:
    index = {
        "date": cfg.date,
        "market": cfg.market,
        "model_path": cfg.model_path,
        "lookback_days": cfg.lookback_days,
        "holdings": [asdict(h) for h in holdings],
        "trading_agents": trading_rows,
        "final_reports": final_reports,
        "latent_result": latent_result,
        "artifacts": {
            "tree_snapshot": str(tree_snapshot) if tree_snapshot and tree_snapshot.exists() else None,
            "tree_narrative": str(tree_narrative) if tree_narrative else None,
            "final_markdown": str(final_markdown),
        },
    }
    destination.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _create_llm(cfg: ExplainabilityConfig, temperature: float) -> Optional[ChatOpenAI]:
    if not cfg.llm:
        return None
    try:
        from langchain_openai import ChatOpenAI as _ChatOpenAI  # type: ignore[import]
    except ImportError as exc:  # noqa: F841
        raise RuntimeError(
            "langchain_openai is required to use LLM features. Install a compatible version or disable --llm."
        ) from exc
    return _ChatOpenAI(model=cfg.llm_model, temperature=temperature, max_retries=2)


def _graph_definition(
    cfg: ExplainabilityConfig,
    out_dir: Path,
    llm_agents: Dict[str, Optional[ChatOpenAI]],
) -> StateGraph:
    graph = StateGraph(PipelineState)

    def node_window(state: PipelineState) -> PipelineState:
        start, end = rolling_window(cfg.date, cfg.lookback_days)
        next_state = dict(state)
        next_state["window"] = (start, end)
        return next_state

    def node_holdings(state: PipelineState) -> PipelineState:
        holdings = collect_holdings(cfg)
        next_state = dict(state)
        next_state["holdings"] = holdings
        return next_state

    def node_tree(state: PipelineState) -> PipelineState:
        start, end = state["window"]
        snapshot = run_explain_tree_module(cfg, start, end, out_dir)
        next_state = dict(state)
        next_state["tree_snapshot"] = snapshot
        return next_state

    def node_tree_narrative(state: PipelineState) -> PipelineState:
        llm = llm_agents.get("tree") if llm_agents else None
        path, markdown = generate_tree_narrative(state["tree_snapshot"], out_dir, llm)
        next_state = dict(state)
        next_state["tree_narrative_path"] = path
        next_state["tree_markdown"] = markdown
        return next_state

    def node_latent(state: PipelineState) -> PipelineState:
        start, end = state["window"]
        latent = run_latent_module(cfg, start, end, out_dir)
        next_state = dict(state)
        next_state["latent_result"] = latent
        return next_state

    def node_trading(state: PipelineState) -> PipelineState:
        rows = run_trading_agents(cfg, state["holdings"], out_dir)
        next_state = dict(state)
        next_state["trading_rows"] = rows
        return next_state

    def node_final(state: PipelineState) -> PipelineState:
        llm = llm_agents.get("final") if llm_agents else None
        final_reports = build_final_reports(
            holdings=state["holdings"],
            trading_rows=state.get("trading_rows", []),
            tree_snapshot=state.get("tree_snapshot"),
            latent_result=state.get("latent_result"),
            llm=llm,
        )
        next_state = dict(state)
        next_state["final_reports"] = final_reports
        return next_state

    graph.add_node("window", node_window)
    graph.add_node("holdings", node_holdings)
    graph.add_node("tree", node_tree)
    graph.add_node("tree_narrative", node_tree_narrative)
    graph.add_node("latent", node_latent)
    graph.add_node("trading", node_trading)
    graph.add_node("final", node_final)

    graph.set_entry_point("window")
    graph.add_edge("window", "holdings")
    graph.add_edge("holdings", "tree")
    graph.add_edge("tree", "tree_narrative")
    graph.add_edge("tree_narrative", "latent")
    graph.add_edge("latent", "trading")
    graph.add_edge("trading", "final")
    graph.add_edge("final", END)

    return graph


def run_explainibility_pipeline(cfg: ExplainabilityConfig) -> PipelineState:
    out_dir = cfg.ensure_output_dir()
    llm_agents = {"tree": _create_llm(cfg, 0.4), "final": _create_llm(cfg, 0.35)}
    graph = _graph_definition(cfg, out_dir, llm_agents).compile()
    initial_state: PipelineState = {}
    final_state = graph.invoke(initial_state)

    tree_text = final_state.get("tree_markdown")
    latent_text = None
    if isinstance(final_state.get("latent_result"), dict):
        latent_text = final_state["latent_result"].get("summary_md")

    final_md_path = out_dir / "explainability_final_results.md"
    write_final_markdown(
        cfg,
        final_state.get("holdings", []),
        tree_text,
        latent_text,
        final_state.get("trading_rows", []),
        final_state.get("final_reports", []),
        final_md_path,
    )

    index_path = out_dir / "orchestrator_index.json"
    write_index_file(
        cfg,
        final_state.get("holdings", []),
        final_state.get("trading_rows", []),
        final_state.get("final_reports", []),
        final_state.get("tree_snapshot"),
        final_state.get("tree_narrative_path"),
        final_state.get("latent_result"),
        final_md_path,
        index_path,
    )

    final_state["final_markdown"] = final_md_path
    final_state["index_path"] = index_path
    return final_state
