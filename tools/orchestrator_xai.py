
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyDUW_zkU6Zz5BEMiDGtEpmqaGVvy-Dr4R0"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRADING_AGENT_ROOT = REPO_ROOT / "trading_agent"
if TRADING_AGENT_ROOT.exists() and str(TRADING_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(TRADING_AGENT_ROOT))

from tools import attention_viz as attention_viz_mod
from tools import explain_tree as explain_tree_mod
from tools import explainability_llm_agent as llm_agent_mod
from tools import explainability_llm_attention as llm_attn_mod
from tradingagents.combined_weight_agent import WeightSynthesisAgent

FINAL_SYSTEM_PROMPT = (
    "You are a senior portfolio strategist. Combine multiple explainability artefacts "
    "(agent summary, decision-tree rules, attention relationships) into one concise note per stock. "
    "Highlight why the model likes the stock, cite key drivers, and flag any risks."
)


@dataclass
class Holding:
    ticker: str
    weight: float
    as_of: Optional[str]


@dataclass
class OrchestratorConfig:
    date: str
    monthly_log_csv: Path
    model_path: str
    market: str
    data_root: str
    top_k: int
    lookback_days: int
    llm: bool
    llm_model: str
    output_dir: Path
    monthly_run_id: Optional[str]


def rolling_window(date_str: str, lookback_days: int) -> tuple[str, str]:
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

def collect_holdings(cfg: OrchestratorConfig) -> List[Holding]:
    print(f"Selecting top {cfg.top_k} holdings for {cfg.date} from {cfg.monthly_log_csv}")
    rows = top_k_for_date_from_log(cfg.monthly_log_csv, cfg.date, top_k=cfg.top_k, run_id=cfg.monthly_run_id)
    holdings = [
        Holding(
            ticker=str(row["ticker"]).strip().upper(),
            weight=float(row["weight"]),
            as_of=row.get("as_of"),
        )
        for row in rows
    ]
    print("Chosen tickers:", [h.ticker for h in holdings])
    return holdings

def run_attention_viz_module(cfg: OrchestratorConfig, start_date: str, end_date: str, out_dir: Path) -> Path:
    debug_log = out_dir / "orchestrator_debug.log"
    with open(debug_log, "a") as f:
        f.write(f"Entering run_attention_viz_module. Date: {cfg.date}\n")
        f.write(f"Module: {attention_viz_mod}\n")
        f.write(f"File: {attention_viz_mod.__file__}\n")

    print(f"Running attention_viz for {start_date} → {end_date} (market={cfg.market})")
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
        "--save-summary",
        "--save-raw",
        "--output-dir",
        str(out_dir),
        "--tickers-csv",
        "tickers.csv",
    ]
    
    with open(debug_log, "a") as f:
        f.write(f"Argv prepared: {argv}\n")
        f.write("Calling attention_viz_mod.main(argv)...\n")

    try:
        import importlib
        import tools.attention_viz
        importlib.reload(tools.attention_viz)
        
        print(f"DEBUG: Calling tools.attention_viz.main from {tools.attention_viz.__file__}")
        print(f"DEBUG: argv = {argv}")
        
        tools.attention_viz.main(argv)
        
        with open(debug_log, "a") as f:
            f.write("Returned from attention_viz_mod.main(argv)\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] attention_viz failed: {exc}")
        import traceback
        traceback.print_exc()
        print(f"[WARN] attention_viz failed: {exc}")
        with open(debug_log, "a") as f:
            f.write(f"Exception in attention_viz_mod.main: {exc}\n")
            import traceback
            traceback.print_exc(file=f)
        (out_dir / "attention_viz_error.txt").write_text(f"Error: {exc}\nArgv: {argv}", encoding="utf-8")
    return out_dir / f"attention_summary_{cfg.market}.json"

def run_explain_tree_module(cfg: OrchestratorConfig, start_date: str, end_date: str, out_dir: Path) -> Path:
    print(f"Running explain_tree for {start_date} → {end_date} (lookback {cfg.lookback_days}d)")
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
        print(f"[WARN] explain_tree failed: {exc}")
    return out_dir / f"explain_tree_{cfg.market}.joblib"

def generate_tree_narrative(cfg: OrchestratorConfig, snapshot: Path, out_dir: Path) -> Optional[Path]:
    if not snapshot.exists():
        print(f"[WARN] Tree snapshot missing at {snapshot}")
        return None
    try:
        ctx = llm_agent_mod.load_snapshot(snapshot)
        prompt = llm_agent_mod.assemble_prompt(ctx)
        (out_dir / "input_tree_llm_prompt.json").write_text(prompt, encoding="utf-8")
        if cfg.llm:
            text = llm_agent_mod.llm_narrative(prompt, model=cfg.llm_model)
        else:
            text = llm_agent_mod.fallback_narrative(ctx)
        destination = out_dir / f"explainability_narrative_tree_{cfg.market}.md"
        destination.write_text(text, encoding="utf-8")
        print(f"Saved tree narrative to {destination}")
        return destination
    except Exception as exc:  
        print(f"[WARN] Tree narrative generation failed: {exc}")
        return None

def generate_attention_narrative(cfg: OrchestratorConfig, summary_path: Path, out_dir: Path) -> Optional[Path]:
    if not summary_path.exists():
        print(f"[WARN] Attention summary missing at {summary_path}")
        return None
    try:
        summary = llm_attn_mod.load_attention_summary(summary_path)
        prompt = llm_attn_mod.assemble_prompt(summary)
        (out_dir / "hgat_attention_prompt.json").write_text(prompt, encoding="utf-8")
        if cfg.llm:
            text = llm_attn_mod.llm_generate(prompt, model=cfg.llm_model)
        else:
            text = f"Loaded HGAT attention summary for model: {summary.get('model_path','N/A')}"
        destination = out_dir / f"hgat_attention_narrative_{cfg.market}.md"
        destination.write_text(text, encoding="utf-8")
        print(f"Saved attention narrative to {destination}")
        return destination
    except Exception as exc: 
        print(f"[WARN] Attention narrative generation failed: {exc}")
        return None

def run_trading_agents(cfg: OrchestratorConfig, holdings: List[Holding], out_dir: Path) -> List[Dict[str, object]]:
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
            print(f"Saved trading agent summary for {holding.ticker} → {out_path}")
        except Exception as exc: 
            rows.append(
                {
                    "ticker": holding.ticker,
                    "success": False,
                    "error": str(exc),
                    "as_of": holding.as_of,
                    "weight": holding.weight,
                }
            )
            print(f"[WARN] Trading agent failed for {holding.ticker}: {exc}")
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

def _load_attention_summary_json(attn_json: Optional[Path]) -> Dict[str, object]:
    if not attn_json or not attn_json.exists():
        return {}
    with attn_json.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def _extract_attention_edges(summary: Dict[str, object], ticker: str) -> List[Dict[str, object]]:
    edges: List[Dict[str, object]] = []
    top_edges = summary.get("top_edges", {}) if isinstance(summary, dict) else {}
    for relation, items in top_edges.items():
        for edge in items:
            src = str(edge.get("source_ticker", ""))
            dst = str(edge.get("target_ticker", ""))
            if ticker in (src, dst):
                edges.append(
                    {
                        "relation": relation,
                        "source": src,
                        "target": dst,
                        "mean_attention": edge.get("mean_attention"),
                    }
                )
    return edges


# Token limit constants (approximate - 1 token ~= 4 chars for English text)
MAX_PROMPT_TOKENS = 100000  # Safe limit for most LLMs
CHARS_PER_TOKEN = 4
MAX_PROMPT_CHARS = MAX_PROMPT_TOKENS * CHARS_PER_TOKEN


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (roughly 4 chars per token)."""
    return len(text) // CHARS_PER_TOKEN


def _truncate_text(text: str, max_chars: int, suffix: str = "\n\n[... truncated ...]") -> str:
    """Truncate text to max_chars, adding suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def _summarize_tree_entry(tree_entry: Optional[Dict[str, object]], max_rules_lines: int = 30) -> Dict[str, object]:
    """Extract only essential information from tree entry to reduce token count."""
    if not tree_entry:
        return {}
    
    summary = {
        "ticker": tree_entry.get("ticker"),
        "r2_score": tree_entry.get("r2_score"),
        "avg_weight": tree_entry.get("avg_weight"),
    }
    
    # Truncate rules to first N lines
    rules = tree_entry.get("rules", "")
    if isinstance(rules, str) and rules.strip():
        lines = rules.strip().splitlines()
        if len(lines) > max_rules_lines:
            summary["rules"] = "\n".join(lines[:max_rules_lines]) + f"\n[... {len(lines) - max_rules_lines} more lines truncated ...]"
        else:
            summary["rules"] = rules
    
    # Keep only top feature importances
    feature_importances = tree_entry.get("feature_importances", {})
    if isinstance(feature_importances, dict):
        # Sort by importance and keep top 10
        sorted_features = sorted(feature_importances.items(), key=lambda x: abs(float(x[1]) if x[1] else 0), reverse=True)[:10]
        summary["top_feature_importances"] = dict(sorted_features)
    elif isinstance(feature_importances, list):
        summary["top_feature_importances"] = feature_importances[:10]
    
    return summary


def _summarize_trading_markdown(markdown: str, max_chars: int = 8000) -> str:
    """Extract key sections from trading markdown and truncate."""
    if not markdown or len(markdown) <= max_chars:
        return markdown
    
    # Try to keep header and summary sections
    lines = markdown.split('\n')
    result_lines = []
    current_chars = 0
    in_summary = False
    
    for line in lines:
        if '## Unified Summary' in line or '## Summary' in line:
            in_summary = True
        elif line.startswith('## ') and in_summary:
            in_summary = False
        
        # Always include headers and summary content
        if line.startswith('#') or in_summary or current_chars < max_chars // 2:
            result_lines.append(line)
            current_chars += len(line) + 1
            
        if current_chars >= max_chars:
            result_lines.append("\n[... content truncated for token limit ...]")
            break
    
    return '\n'.join(result_lines)


def _call_final_llm(prompt: str, model: str) -> str:
    import google.generativeai as genai

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY missing for final LLM synthesis")
    genai.configure(api_key=key)
    
    # Check and truncate prompt if needed
    estimated_tokens = _estimate_tokens(prompt)
    if estimated_tokens > MAX_PROMPT_TOKENS:
        print(f"[WARN] Prompt has ~{estimated_tokens} tokens, truncating to ~{MAX_PROMPT_TOKENS}")
        prompt = _truncate_text(prompt, MAX_PROMPT_CHARS)
    
    llm = genai.GenerativeModel(model_name=model, system_instruction=FINAL_SYSTEM_PROMPT)
    resp = llm.generate_content(prompt, generation_config={"temperature": 0.35, "top_p": 0.9})
    text = getattr(resp, "text", None)
    if not text:
        raise RuntimeError("Final LLM returned empty response")
    return text.strip()


def _final_prompt_payload(
    *,
    ticker: str,
    as_of: Optional[str],
    weight: Optional[float],
    trading_markdown: str,
    tree_entry: Optional[Dict[str, object]],
    attention_edges: List[Dict[str, object]],
    max_total_chars: int = MAX_PROMPT_CHARS,
) -> str:
    """Build prompt payload with token limit enforcement."""
    
    # Summarize/truncate each component
    summarized_tree = _summarize_tree_entry(tree_entry, max_rules_lines=25)
    summarized_markdown = _summarize_trading_markdown(trading_markdown, max_chars=6000)
    
    # Limit attention edges
    limited_edges = attention_edges[:20] if attention_edges else []
    
    payload = {
        "ticker": ticker,
        "as_of": as_of,
        "weight": weight,
        "trading_agent_summary": summarized_markdown,
        "tree_rules_summary": summarized_tree,
        "attention_edges": limited_edges,
    }
    
    result = json.dumps(payload, indent=2, default=str)
    
    # Final safety truncation
    if len(result) > max_total_chars:
        print(f"[WARN] Final prompt payload ({len(result)} chars) exceeds limit, truncating")
        result = _truncate_text(result, max_total_chars)
    
    return result


def _fallback_final_summary(ticker: str, trading_points: List[str], tree_entry: Optional[Dict[str, object]]) -> str:
    pieces = [f"Fallback summary for {ticker}."]
    if trading_points:
        pieces.append("Key takeaways: " + "; ".join(trading_points))
    if tree_entry:
        rules = tree_entry.get("rules")
        if isinstance(rules, str) and rules.strip():
            pieces.append("Tree rules snippet: " + rules.strip().splitlines()[0])
    return " ".join(pieces)


def build_final_reports(
    *,
    summary_rows: List[Dict[str, object]],
    tree_snapshot: Optional[Path],
    attn_summary_path: Optional[Path],
    llm: bool,
    llm_model: str,
    per_stock_tree: Optional[Dict[str, Dict[str, object]]] = None,
    attn_summary: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    per_stock_tree = per_stock_tree if per_stock_tree is not None else _load_tree_per_stock(tree_snapshot)
    attn_summary = attn_summary if attn_summary is not None else _load_attention_summary_json(attn_summary_path)
    results: List[Dict[str, object]] = []

    for item in summary_rows:
        ticker = str(item.get("ticker", "?")).upper()
        if not item.get("success"):
            results.append({"ticker": ticker, "success": False, "error": item.get("error", "unknown")})
            continue
        report_path = Path(item["output_path"])
        try:
            trading_md = report_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            trading_md = ""
        tree_entry = per_stock_tree.get(ticker)
        attention_edges = _extract_attention_edges(attn_summary, ticker)
        prompt = _final_prompt_payload(
            ticker=ticker,
            as_of=item.get("as_of"),
            weight=item.get("weight"),
            trading_markdown=trading_md,
            tree_entry=tree_entry,
            attention_edges=attention_edges,
        )
        try:
            if llm:
                text = _call_final_llm(prompt, llm_model)
                used_llm = True
            else:
                text = _fallback_final_summary(ticker, item.get("summary_points") or [], tree_entry)
                used_llm = False
        except Exception as exc:  # noqa: BLE001
            text = f"[FINAL LLM ERROR] {exc}. Fallback: " + _fallback_final_summary(
                ticker, item.get("summary_points") or [], tree_entry
            )
            used_llm = False
        results.append({"ticker": ticker, "success": True, "output": text, "llm_used": used_llm})
    return results


def _read_text(path: Optional[Path]) -> Optional[str]:
    if path and path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def render_dashboard(
    *,
    cfg: OrchestratorConfig,
    holdings: List[Holding],
    trading_rows: List[Dict[str, object]],
    final_reports: List[Dict[str, object]],
    tree_text: Optional[str],
    attention_text: Optional[str],
) -> None:
    for rank, holding in enumerate(holdings, start=1):
        print(f"{rank:02d}. {holding.ticker:<8} | weight {holding.weight*100:.2f}% | as_of {holding.as_of or 'n/a'}")

    if tree_text:
        print("Tree Narrative (full text saved to disk):")
        print(tree_text)
    if attention_text:
        print("\nAttention Narrative:")
        print(attention_text)

    trading_map = {row.get("ticker"): row for row in trading_rows}
    final_map = {row.get("ticker"): row for row in final_reports}

    for holding in holdings:
        trade = trading_map.get(holding.ticker)
        final = final_map.get(holding.ticker)
        print(f"\n>>> {holding.ticker} — target {holding.weight*100:.2f}%")
        if trade and trade.get("success"):
            for point in trade.get("summary_points") or []:
                print(f"   - {point}")
            print(f"   report: {trade.get('output_path')}")
        elif trade:
            print(f"   trading agent failed: {trade.get('error','unknown')}")
        else:
            print("   trading agent output missing")

        if final and final.get("success"):
            tag = "LLM" if final.get("llm_used") else "fallback"
            print(f"   final summary ({tag}):\n{final.get('output')}")
        elif final:
            print(f"   final synthesis failed: {final.get('error','unknown')}")
        else:
            print("   final synthesis missing")


def write_final_markdown(
    cfg: OrchestratorConfig,
    holdings: List[Holding],
    tree_text: Optional[str],
    attention_text: Optional[str],
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

    lines.append("\n## Attention Narrative")
    lines.append(attention_text or "(attention narrative unavailable)")

    lines.append("\n## Per-Stock Insights")
    for holding in holdings:
        lines.append(f"\n### {holding.ticker}")
        trade = trading_map.get(holding.ticker)
        if trade and trade.get("success"):
            lines.append("**Trading Agent Highlights**")
            for point in trade.get("summary_points") or []:
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
    print(f"Final explainability markdown saved to {destination}")


def write_index_file(
    cfg: OrchestratorConfig,
    holdings: List[Holding],
    trading_rows: List[Dict[str, object]],
    final_reports: List[Dict[str, object]],
    tree_snapshot: Optional[Path],
    tree_narrative: Optional[Path],
    attention_summary: Optional[Path],
    attention_narrative: Optional[Path],
    final_markdown: Path,
    destination: Path,
) -> None:
    index = {
        "date": cfg.date,
        "market": cfg.market,
        "model_path": cfg.model_path,
        "lookback_days": cfg.lookback_days,
        "holdings": [holding.__dict__ for holding in holdings],
        "trading_agents": trading_rows,
        "final_reports": final_reports,
        "artifacts": {
            "tree_snapshot": str(tree_snapshot) if tree_snapshot and tree_snapshot.exists() else None,
            "tree_narrative": str(tree_narrative) if tree_narrative else None,
            "attention_summary": str(attention_summary) if attention_summary and attention_summary.exists() else None,
            "attention_narrative": str(attention_narrative) if attention_narrative else None,
            "final_markdown": str(final_markdown),
        },
    }
    destination.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Index written to {destination}")


def run_orchestrator(cfg: OrchestratorConfig) -> None:
    out_dir = cfg.output_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    holdings = collect_holdings(cfg)
    window_start, window_end = rolling_window(cfg.date, cfg.lookback_days)

    attention_summary_path = run_attention_viz_module(cfg, window_start, window_end, out_dir)
    tree_snapshot_path = run_explain_tree_module(cfg, window_start, window_end, out_dir)

    tree_narrative_path = generate_tree_narrative(cfg, tree_snapshot_path, out_dir)
    attention_narrative_path = generate_attention_narrative(cfg, attention_summary_path, out_dir)

    trading_rows = run_trading_agents(cfg, holdings, out_dir)

    per_stock_tree = _load_tree_per_stock(tree_snapshot_path)
    attn_summary_payload = _load_attention_summary_json(attention_summary_path)
    final_reports = build_final_reports(
        summary_rows=trading_rows,
        tree_snapshot=tree_snapshot_path,
        attn_summary_path=attention_summary_path,
        llm=cfg.llm,
        llm_model=cfg.llm_model,
        per_stock_tree=per_stock_tree,
        attn_summary=attn_summary_payload,
    )

    tree_text = _read_text(tree_narrative_path)
    attention_text = _read_text(attention_narrative_path)

    render_dashboard(
        cfg=cfg,
        holdings=holdings,
        trading_rows=trading_rows,
        final_reports=final_reports,
        tree_text=tree_text,
        attention_text=attention_text,
    )

    final_md_path = out_dir / "explainability_final_results.md"
    write_final_markdown(
        cfg=cfg,
        holdings=holdings,
        tree_text=tree_text,
        attention_text=attention_text,
        trading_rows=trading_rows,
        final_reports=final_reports,
        destination=final_md_path,
    )

    index_path = out_dir / "orchestrator_index.json"
    write_index_file(
        cfg=cfg,
        holdings=holdings,
        trading_rows=trading_rows,
        final_reports=final_reports,
        tree_snapshot=tree_snapshot_path,
        tree_narrative=tree_narrative_path,
        attention_summary=attention_summary_path,
        attention_narrative=attention_narrative_path,
        final_markdown=final_md_path,
        destination=index_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SmartFolio explainability orchestrator")
    parser.add_argument("--date", required=True, help="Snapshot date to analyze (YYYY-MM-DD)")
    parser.add_argument("--monthly-log-csv", required=True, help="Path to monthly final_test_weights CSV")
    parser.add_argument("--monthly-run-id", default=None, help="Optional run_id filter when CSV tracks multiple runs")
    parser.add_argument("--model-path", required=True, help="Path to trained PPO checkpoint")
    parser.add_argument("--market", default="hs300")
    parser.add_argument("--data-root", default="dataset_default")
    parser.add_argument("--top-k", type=int, default=5, help="Number of holdings to explain (default 5)")
    parser.add_argument("--lookback-days", type=int, default=60, help="Days before --date for explainability datasets")
    parser.add_argument("--llm", action="store_true", help="Enable LLM generation for narratives")
    parser.add_argument("--llm-model", default="gemini-2.0-flash")
    parser.add_argument("--output-dir", default="explainability_results")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = OrchestratorConfig(
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
    )
    run_orchestrator(cfg)
if __name__ == "__main__":
    main()