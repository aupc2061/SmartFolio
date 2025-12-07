from __future__ import annotations
import argparse, json, os, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import numpy as np
import joblib
import google.generativeai as genai

# Token limit constants
MAX_PROMPT_TOKENS = 100000  # Safe limit for most LLMs
CHARS_PER_TOKEN = 4
MAX_PROMPT_CHARS = MAX_PROMPT_TOKENS * CHARS_PER_TOKEN
MAX_RULES_LINES_PER_STOCK = 20
MAX_FEATURE_IMPORTANCES = 8
MAX_STOCKS_IN_PROMPT = 10

SYSTEM_PROMPT = ("""
You are a Financial Analyst writing a simplified newsletter for retail investors. Your goal is to explain an AI's trading strategy in plain English, removing all technical jargon.

### Input Data
You will receive "Decision Rules" that look like this:
`If AARTIIND.NS::Volume > 2.0 AND BATAINDIA.NS::Close < -1.0`...

### Interpretation Guide (Z-Scores)
The input data uses Z-scores (normalized values), not raw prices:
- **Volume > 1**: High trading volume / Activity spike.
- **Volume < -1**: Low trading volume / Quiet.
- **Close > 1**: Price is high relative to recent history (Uptrend).
- **Close < -1**: Price is low (Downtrend / "Buying the Dip").
- **Returns > 0**: Positive momentum.

### Your Task
Translate the math into a story.
1.  **Identify the Vibe**: Is the AI chasing momentum (buying high)? Is it contrarian (buying the dip)?
2.  **Explain Relationships**: If the AI buys Stock A based on Stock B's movement, call it a "Cross-market signal" or "Sector correlation".

### Output Rules (Strict)
1.  **NO JARGON**: Do not use words like "thresholds", "nodes", "z-scores", "coefficients", "average weight".
2.  **NO RAW NUMBERS**: Do not quote the specific values (like `4.42` or `0.51`). Use descriptive terms like "significant spike", "moderate drop", "stable".
3.  **NO "AVERAGE WEIGHT"**: Do not mention the allocation percentages.
4.  **Short & Simple**: Use short sentences.

### Output Format
For each stock, provide exactly this structure:

**[Stock Name]**
* **The Strategy**: [A catchy 3-6 word summary, e.g., "Momentum Play backed by Sector Peers"]
* **The Logic**: [A plain-English paragraph explaining *why*. E.g., "The model is buying this stock because it sees a massive volume spike in its peer, Tata Steel. It seems to be hedging against weakness in the broader market, only buying when the sector is quiet."]
* **Key Signals to Watch**:
    * [Related Stock]: [What to look for, e.g., "High Volume"]
    * [Related Stock]: [What to look for, e.g., "Price Drop"]

""")


@dataclass
class SnapshotContext:
    metadata: Dict[str, str]
    filtered_data: Dict[str, object]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SmartFolio explainability narrative (Gemini 2.0 Flash).")
    p.add_argument("--snapshot", default="explainability_results/explain_tree_custom.joblib")
    p.add_argument("--llm", action="store_true", help="Enable Gemini generation")
    p.add_argument("--llm-model", default="gemini-2.0-flash", help="LLM model name")
    p.add_argument("--output", default="explainability_results/explainability_narrative.md")
    p.add_argument("--print", action="store_true")
    return p.parse_args()

def safe_convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def convert_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(v) for v in obj]
    else:
        return safe_convert(obj)


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (roughly 4 chars per token)."""
    return len(text) // CHARS_PER_TOKEN


def _truncate_text(text: str, max_chars: int, suffix: str = "\n[... truncated ...]") -> str:
    """Truncate text to max_chars, adding suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def _summarize_per_stock_entry(entry: Dict, max_rules_lines: int = MAX_RULES_LINES_PER_STOCK) -> Dict:
    """Summarize a single stock's tree data to reduce token count."""
    summary = {
        "ticker": entry.get("ticker"),
        "r2_score": entry.get("r2_score"),
        "avg_weight": entry.get("avg_weight"),
    }
    
    # Truncate rules to first N lines
    rules = entry.get("rules", "")
    if isinstance(rules, str) and rules.strip():
        lines = rules.strip().splitlines()
        if len(lines) > max_rules_lines:
            summary["rules"] = "\n".join(lines[:max_rules_lines]) + f"\n[... {len(lines) - max_rules_lines} more rule lines omitted ...]"
        else:
            summary["rules"] = rules
    
    # Keep only top feature importances
    feature_importances = entry.get("feature_importances", {})
    if isinstance(feature_importances, dict):
        sorted_features = sorted(
            feature_importances.items(), 
            key=lambda x: abs(float(x[1]) if x[1] else 0), 
            reverse=True
        )[:MAX_FEATURE_IMPORTANCES]
        summary["top_features"] = dict(sorted_features)
    elif isinstance(feature_importances, list):
        summary["top_features"] = feature_importances[:MAX_FEATURE_IMPORTANCES]
    
    return summary


def _summarize_filtered_data(data: Dict) -> Dict:
    """Create a token-efficient summary of the explainability data."""
    summarized = {}
    
    # Copy simple scalar values
    for key in ["global_r2", "X_shape", "Y_shape"]:
        if key in data:
            summarized[key] = data[key]
    
    # Summarize avg_weights (keep top stocks only)
    avg_weights = data.get("avg_weights", {})
    if isinstance(avg_weights, dict):
        sorted_weights = sorted(avg_weights.items(), key=lambda x: float(x[1]) if x[1] else 0, reverse=True)
        summarized["top_avg_weights"] = dict(sorted_weights[:MAX_STOCKS_IN_PROMPT])
    
    # Summarize per_stock (the main data that causes token explosion)
    per_stock = data.get("per_stock", {})
    if isinstance(per_stock, dict):
        # Sort by avg_weight and take top stocks
        stock_items = list(per_stock.items())
        stock_items.sort(
            key=lambda x: float(x[1].get("avg_weight", 0)) if isinstance(x[1], dict) else 0,
            reverse=True
        )
        
        summarized_per_stock = {}
        for ticker, entry in stock_items[:MAX_STOCKS_IN_PROMPT]:
            if isinstance(entry, dict):
                summarized_per_stock[ticker] = _summarize_per_stock_entry(entry)
        
        summarized["per_stock"] = summarized_per_stock
        
        if len(stock_items) > MAX_STOCKS_IN_PROMPT:
            summarized["_note"] = f"Showing top {MAX_STOCKS_IN_PROMPT} of {len(stock_items)} stocks by weight"
    
    return summarized


def load_snapshot(path: Path) -> SnapshotContext:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    data = joblib.load(path)
    if not isinstance(data, dict):
        raise TypeError("Joblib file must contain a dict.")

    keep = ["per_stock", "avg_weights", "top_indices", "global_r2", "X_shape", "Y_shape"]
    filtered = {k: data.get(k) for k in keep if k in data}

    meta = {"model_path": str(path), "included_keys": list(filtered.keys())}
    return SnapshotContext(metadata=meta, filtered_data=filtered)


def assemble_prompt(ctx: SnapshotContext) -> str:
    """
    Builds a structured Gemini-optimized prompt emphasizing narrative over data.
    Token-limited to prevent context length errors.
    """

    example_block = (
        "Example output:\n\n"
        "**ICICIBANK.NS**\n"
        "* **The Strategy**: Sector Rotation based on Peers\n"
        "* **The Logic**: The model is aggressively buying ICICI Bank because it sees a major volume spike in SBI. It uses this as a confirmation signal for the entire banking sector. It avoids this trade if Infosys is also crashing, treating that as a broader market risk.\n"
        "* **Key Signals to Watch**:\n"
        "    * SBIN.NS: High Volume\n"
        "    * INFY.NS: Price Drop\n\n"
    )

    instructions = (
        "You are given structured JSON data summarizing decision-tree surrogates. "
        "Feature names follow the pattern 'TICKER::t-<N>::Metric', where t-0 is the most recent trading day, t-1 is one day back, etc. "
        "Convert the 'rules' and 'feature_importances' for each stock into the plain English format defined in the system prompt. "
        "Ignore the 'avg_weight' field completely in your output.\n"
    )

    # Summarize the data to reduce token count
    summarized_data = _summarize_filtered_data(ctx.filtered_data)

    payload = {
        "system_instruction": SYSTEM_PROMPT,
        "metadata": ctx.metadata,
        "instructions": instructions,
        "example_output": example_block,
        "explainability_data": summarized_data,
    }
    
    result = json.dumps(convert_keys_to_str(payload), indent=2, default=safe_convert)
    
    # Check token count and warn/truncate if needed
    estimated_tokens = _estimate_tokens(result)
    print(f"[INFO] Assembled prompt: ~{estimated_tokens} tokens ({len(result)} chars)")
    
    if estimated_tokens > MAX_PROMPT_TOKENS:
        print(f"[WARN] Prompt exceeds {MAX_PROMPT_TOKENS} token limit, truncating...")
        result = _truncate_text(result, MAX_PROMPT_CHARS)
        print(f"[INFO] Truncated to {len(result)} chars (~{_estimate_tokens(result)} tokens)")

    return result

def llm_narrative(prompt: str, model="gemini-2.0-flash", retries=3, delay=4) -> str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GOOGLE_API_KEY / GEMINI_API_KEY.")
    genai.configure(api_key=key)
    
    # Final token check before sending to LLM
    estimated_tokens = _estimate_tokens(prompt)
    if estimated_tokens > MAX_PROMPT_TOKENS:
        print(f"[WARN] Prompt has ~{estimated_tokens} tokens, applying final truncation")
        prompt = _truncate_text(prompt, MAX_PROMPT_CHARS)
    
    llm = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_PROMPT)

    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Gemini 2.0 Flash call attempt {attempt}/{retries}")
            resp = llm.generate_content(prompt, generation_config={"temperature": 0.4, "top_p": 0.9})
            text = getattr(resp, "text", None)
            if text:
                return text.strip()
            raise RuntimeError("Empty Gemini response")
        except Exception as e:
            msg = str(e)
            if "429" in msg and attempt < retries:
                print(f"[WARN] Rate limited (429). Retrying in {delay}s…")
                time.sleep(delay)
                continue
            # Check if it's a token limit error and try with more aggressive truncation
            if "token" in msg.lower() and "limit" in msg.lower() and attempt < retries:
                print(f"[WARN] Token limit exceeded, reducing prompt size and retrying...")
                prompt = _truncate_text(prompt, MAX_PROMPT_CHARS // 2)
                continue
            print(f"[ERROR] Gemini call failed: {e}")
            return f"**LLM generation failed:** {e}"
    return "**LLM generation unavailable.**"

def fallback_narrative(ctx: SnapshotContext) -> str:
    d = ctx.filtered_data
    return (
        f"Fallback summary — global R² {d.get('global_r2','n/a')}, "
        f"{len(d.get('per_stock',{}))} stocks analyzed. "
        "Detailed logic requires LLM generation."
    )

def main() -> None:
    args = parse_args()
    snap = Path(args.snapshot).expanduser()
    try:
        ctx = load_snapshot(snap)
        print(f"[INFO] Loaded keys: {ctx.metadata['included_keys']}")
    except Exception as e:
        print(f"[ERROR] Snapshot load failed: {e}")
        sys.exit(1)

    prompt = assemble_prompt(ctx)
    prompt_path = Path(args.output).with_name("input_tree_llm_prompt.json")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"[INFO] Prompt saved → {prompt_path}")

    if args.llm:
        output_text = llm_narrative(prompt, model=args.llm_model)
    else:
        output_text = fallback_narrative(ctx)

    out_path = Path(args.output)
    out_path.write_text(output_text, encoding="utf-8")

    if args.print:
        print("\n--- Generated Narrative ---\n")
        print(output_text)
    print(f"Narrative written to {out_path}")


if __name__ == "__main__":
    main()
