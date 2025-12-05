#!/usr/bin/env python3
"""
Latent Factor Explanation Module (LLM-Enhanced).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add repo root so `python explainibility_agents/explain_latent.py` works
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.latent_factors import collect_traces, train_ae, extract_features
from explainibility_agents.tradingagents.llm_provider import ProviderError, generate_completion

def _normalize_symbol(token: str) -> str:
    token = token.strip()
    return token.replace(".ns", "").replace(".NS", "").upper()


def _clean_feature_name(name: str) -> str:
    cleaned = name.replace("t-0", "").replace("t-1", "Lag-1").replace("_", " ").strip()

    def _replace_symbols(match: re.Match[str]) -> str:
        inside = match.group(1)
        parts = [part for part in (p.strip() for p in inside.split(",")) if part]
        normalized = ", ".join(_normalize_symbol(part) for part in parts)
        return f"[{normalized}]"

    cleaned = re.sub(r"\[([^\]]+)\]", _replace_symbols, cleaned)
    cleaned = cleaned.replace(".ns", ".NS")  # normalize suffix before casing
    words = cleaned.split()
    return " ".join(word.capitalize() if word.isalpha() else word for word in words)


def _extract_primary_symbol(feature: str) -> Optional[str]:
    bracket_match = re.search(r"\[([^\]]+)\]", feature)
    if bracket_match:
        tokens = [tok for tok in (p.strip() for p in bracket_match.group(1).split(",")) if tok]
        if tokens:
            return _normalize_symbol(tokens[0])
    if "::" in feature:
        prefix = feature.split("::", maxsplit=1)[0]
        return _normalize_symbol(prefix)
    return None

def _get_llm_interpretations(factors: List[tuple], model_name: str) -> Dict[str, str]:
    """Use the centralized LLM provider to interpret latent factors."""

    if not factors:
        return {}

    prompt_lines = [
        "You are a financial analyst analyzing the latent space of an RL trading agent.",
        "Each factor contains its top weighted drivers (feature + weight).",
        "For every factor, return a concise 3-5 word label describing the learned market concept.",
        "Respond with valid JSON mapping the factor id to the label.",
        "\nFactors to Analyze:",
    ]
    for fid, info in factors:
        drivers = ", ".join(
            f"{_clean_feature_name(n)} ({w:.2f})" for n, w in zip(info["feature_names"][:4], info["weights"][:4])
        )
        prompt_lines.append(f"- {fid}: {drivers}")

    try:
        response = generate_completion(
            "\n".join(prompt_lines),
            system_prompt="Name latent financial factors based on their drivers.",
            model=model_name,
            temperature=0.2,
            top_p=0.8,
        )
        return json.loads(response)
    except (ProviderError, json.JSONDecodeError) as exc:
        print(f"[WARN] LLM interpretation failed: {exc}")
        return {}

def _derive_interpretation(features: List[str]) -> str:
    """Fallback heuristic if LLM is disabled or fails."""
    if not features: return "Unknown"
    first_parts = [f.split('_')[0] for f in features[:3]]
    last_parts = [f.split('_')[-1] for f in features[:3]]
    primary = _extract_primary_symbol(features[0])
    if len(set(first_parts)) == 1 and primary:
        return f"Single Stock: {primary}"
    if len(set(last_parts)) == 1: return f"Market-wide {last_parts[0].capitalize()}"
    return "Composite Factor"


def _prettify_label(label: str) -> str:
    if not label:
        return "Unlabeled Factor"
    clean = label.strip()
    if clean.lower().startswith("single stock:"):
        _, _, ticker = clean.partition(":")
        ticker = ticker.strip().upper() or "N/A"
        return f"Single Stock · {ticker}"
    return clean[:1].upper() + clean[1:]


def _describe_factor(label: str, features: List[str]) -> str:
    pretty = _prettify_label(label)
    top_names = [_clean_feature_name(name) for name in features[:2] if name]
    driver_stub = ", ".join(top_names) if top_names else "its strongest drivers"
    pretty_lower = pretty.lower()

    if "single stock" in pretty_lower:
        focus = pretty.split("·", maxsplit=1)[-1].strip() or "the focus ticker"
        return f"Concentrated exposure to {focus}, primarily reacting to {driver_stub}."
    if "market-wide" in pretty_lower:
        regime = pretty.split(" ", maxsplit=1)[-1].strip().lower()
        return f"Broad {regime} regime captured through {driver_stub}."
    if "composite" in pretty_lower:
        return f"Blended multi-name signal mixing {driver_stub}."
    return f"{pretty} pattern led by {driver_stub}."

def run_latent_pipeline(
    model_path: str,
    market: str,
    start_date: str,
    end_date: str,
    data_root: str,
    output_dir: Path,
    device: str = "cpu",
    llm: bool = False,
    llm_model: str = "gpt-4.1-mini"
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect Traces
    traces_name = "local_traces"
    traces_path = output_dir / f"{traces_name}.npz"
    if not traces_path.exists():
        print(f"[Latent] Collecting traces {start_date} -> {end_date}...")
        collect_traces.main([
            "--model-path", model_path, "--market", market,
            "--test-start-date", start_date, "--test-end-date", end_date,
            "--data-root", data_root, "--device", device,
            "--output-dir", str(output_dir), "--output-name", traces_name,
        ])

    # 2. Train SAE
    checkpoint_dir = output_dir / "checkpoints"
    sae_ckpt = checkpoint_dir / "sparse_ae_best.pt"
    if not sae_ckpt.exists():
        print(f"[Latent] Training Sparse Autoencoder...")
        train_ae.main([
            "--data-path", str(traces_path), "--output-dir", str(checkpoint_dir),
            "--model-type", "topk", "--latent-dim", "64", "--k", "8",
            "--epochs", "20", "--batch-size", "256", "--device", device, "--save-best"
        ])
        sae_ckpt = checkpoint_dir / "sparse_ae_best.pt"

    # 3. Extract Features
    analysis_dir = output_dir / "analysis"
    top_features_path = analysis_dir / "top_features.json"
    if not top_features_path.exists():
        print(f"[Latent] Extracting feature interpretations...")
        extract_features.main([
            "--checkpoint", str(sae_ckpt), "--data-path", str(traces_path),
            "--output-dir", str(analysis_dir), "--device", device, "--top-k", "5"
        ])

    # 4. Generate Dashboard
    with open(top_features_path, "r") as f:
        data = json.load(f)

    # Sort top 8 factors by strength
    sorted_items = sorted(
        data["top_features"].items(), 
        key=lambda x: max([abs(w) for w in x[1]["weights"]]) if x[1]["weights"] else 0, 
        reverse=True
    )[:8]

    # Get LLM Interpretations if requested
    llm_labels = {}
    if llm:
        print(f"[Latent] Asking {llm_model} to interpret factor semantics...")
        llm_labels = _get_llm_interpretations(sorted_items, llm_model)

    lines = []
    lines.append("### Latent Factor Analysis")
    lines.append(f"**Analysis Window:** {start_date} to {end_date}")
    lines.append("")
    lines.append("| Top Drivers | Interpretation |")
    lines.append("| :--- | :--- |")
    
    for latent_id, info in sorted_items:
        feats = info["feature_names"]
        weights = info["weights"]
        
        driver_strs = []
        for n, w in zip(feats[:3], weights[:3]):
            driver_strs.append(f"- **{_clean_feature_name(n)}** ({w:.2f})")
        
        # Use LLM label if available, else heuristic
        clean_id = latent_id.replace("latent_", "F")
        label = llm_labels.get(latent_id) or llm_labels.get(clean_id) or _derive_interpretation(feats)
        pretty_label = _prettify_label(label)
        drivers_block = "<br>".join([f"**{pretty_label}**"] + driver_strs)
        interpretation = _describe_factor(label, feats)
        lines.append(f"| {drivers_block} | {interpretation} |")

    lines.append("")
    lines.append("> *Labels generated by AI analysis of feature weights.*" if llm else "> *Labels generated by heuristic rules.*")

    summary_md = "\n".join(lines)
    (output_dir / "latent_summary.md").write_text(summary_md, encoding="utf-8")

    return {"success": True, "summary_md": summary_md, "artifacts_dir": str(output_dir)}

def main():
    # ... (Same argument parser as before) ...
    pass

if __name__ == "__main__":
    main()