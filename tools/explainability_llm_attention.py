"""
SmartFolio HGAT Attention-Only Explainability
==============================================

This script reads a JSON file that contains only the "attention_summary"
key and produces a Gemini 2.0 Flash–optimized interpretation.

It explains:
- Semantic attention distributions
- Top-edge (industry, positive, negative) structures
- Portfolio implications

Output: Markdown narrative saved to `explainability_results/hgat_attention_narrative.md`
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import google.generativeai as genai

# Token limit constants
MAX_PROMPT_TOKENS = 100000
CHARS_PER_TOKEN = 4
MAX_PROMPT_CHARS = MAX_PROMPT_TOKENS * CHARS_PER_TOKEN
MAX_EDGES_PER_RELATION = 15


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (roughly 4 chars per token)."""
    return len(text) // CHARS_PER_TOKEN


def _truncate_text(text: str, max_chars: int, suffix: str = "\n[... truncated ...]") -> str:
    """Truncate text to max_chars, adding suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


SYSTEM_PROMPT = (
    "You are a quantitative explainability analyst specializing in graph-based models. "
    "You will analyze the HGAT (Hierarchical Graph Attention Network) explainability summary "
    "used within a reinforcement-learning portfolio policy. "
    "Your goal is to interpret semantic attention weights, inter-stock relationships, "
    "and cross-stock influence patterns clearly and concisely for a technical audience."
)


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text."""
    return len(text) // CHARS_PER_TOKEN


def _truncate_text(text: str, max_chars: int, suffix: str = "\n[... truncated ...]") -> str:
    """Truncate text to max_chars."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix

def parse_args():
    p = argparse.ArgumentParser(description="Explain HGAT attention summary JSON using Gemini 2.0 Flash.")
    p.add_argument(
        "--attention-json",
        required=True,
        help="Path to JSON file containing only 'attention_summary'.",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        help="Enable Gemini 2.0 Flash for generating the explanation.",
    )
    p.add_argument(
        "--llm-model",
        default="gemini-2.0-flash",
        help="Gemini model name (default: gemini-2.0-flash).",
    )
    p.add_argument(
        "--output",
        default="explainability_results/hgat_attention_narrative.md",
        help="Path to save the final narrative.",
    )
    p.add_argument(
        "--print",
        action="store_true",
        help="Print the narrative output to console.",
    )
    return p.parse_args()

def load_attention_summary(attention_path: Path):
    if not attention_path.exists():
        raise FileNotFoundError(f"Attention summary file not found: {attention_path}")

    with open(attention_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "attention_summary" in data:
        attn = data["attention_summary"]
    else:
        attn = data

    # Limit top_edges to reduce token count
    top_edges = attn.get("top_edges", {})
    limited_edges = {}
    for relation, edges in top_edges.items():
        if isinstance(edges, list):
            limited_edges[relation] = edges[:MAX_EDGES_PER_RELATION]
        else:
            limited_edges[relation] = edges

    # compress large arrays for readability
    summary = {
        "model_path": attn.get("model_path", ""),
        "market": attn.get("market", ""),
        "num_stocks": attn.get("num_stocks", ""),
        "semantic_labels": attn.get("semantic_labels", []),
        "semantic_mean": attn.get("semantic_mean", []),
        "top_edges": limited_edges,
    }

    if "avg_allocations" in attn:
        arr = np.array(attn["avg_allocations"])
        summary["avg_allocations_summary"] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return summary

def assemble_prompt(attn_summary: dict) -> str:
    """
    Builds a precise, LLM-ready prompt for HGAT interpretability.
    """
    instructions = (
        "### Task\n"
        "You are explaining model-driven portfolio decisions to retail investors and wealth managers.\n"
        "Translate this HGAT attention summary into clear, actionable insights.\n\n"
        "### Translation Guide\n"
        "- 'Positive attention' → 'Momentum relationships' (stocks that tend to rise together)\n"
        "- 'Negative attention' → 'Hedging relationships' (stocks used to offset risk)\n"
        "- 'Industry attention' → 'Sector linkages' (stocks grouped by industry correlation)\n"
        "- 'Self attention' → 'Independent stock behavior' (stock's own price history)\n\n"
        "### Required Sections\n"
        "1. **Portfolio Strategy Summary** (2-3 sentences)\n"
        "   - What is the model's overall approach? (momentum-focused? sector-balanced? hedged?)\n"
        "   - Translate semantic weights into plain English percentages\n"
        "   - Example: 'Our model focuses 35% on momentum relationships, 24% on hedging, 22% on sector trends, and 19% on individual stock behavior.'\n\n"
        "2. **Key Stock Relationships** (3-4 bullet points)\n"
        "   - Identify THE most important stock (highest attention target)\n"
        "   - Explain its role: Is it a core holding? A hedge? A sector proxy?\n"
        "   - List 2-3 stocks influencing it and WHY (momentum boost? risk offset?)\n"
        "   - Example: 'OBEROIRLTY.NS is the portfolio anchor, receiving strong momentum signals from CDSL.NS and PVRINOX.NS (both consumer discretionary stocks).'\n\n"
        "3. **Risk & Concentration Warnings** (2-3 bullet points)\n"
        "   - Flag if >40% attention goes to 1-2 stocks (concentration risk)\n"
        "   - Identify sector clustering risks (e.g., 'Heavy FMCG exposure through MARICO.NS-BRITANNIA.NS linkage')\n"
        "   - Note hedging adequacy: Is negative attention spread or concentrated?\n\n"
        "### Tone & Style\n"
        "- Write for a smart investor, not a data scientist\n"
        "- Use specific stock names, not index numbers\n"
        "- Quantify everything (percentages, attention weights as risk scores)\n"
        "- Be honest about limitations ('This shows correlation, not causation')\n"
        "- NEVER mention 'AI', 'LLM', or 'artificial intelligence'—always say 'our model', 'based on predictions', 'the portfolio strategy'\n"
        "- NO JARGON: No 'graph neural networks', 'heterogeneous attention', 'semantic fusion'\n"
    )

    example_block = (
        "Example Output:\n\n"
        "## Portfolio Strategy Summary\n"
        "Our model employs a momentum-heavy strategy, allocating 43% of its decision-making to identifying stocks that rise together, "
        "20% to sector correlations, 24% to hedging relationships, and 13% to individual stock patterns. "
        "This indicates a growth-oriented approach that seeks to amplify returns through correlated holdings.\n\n"
        "## Key Stock Relationships\n"
        "- **OBEROIRLTY.NS is the portfolio's primary driver**, receiving the strongest momentum signals from CDSL.NS, PVRINOX.NS, and AARTIIND.NS. "
        "These stocks historically move in sync during bull markets, suggesting the strategy expects continued economic expansion.\n"
        "- **MARICO.NS serves as the main hedge**, with negative correlations to POONAWALLA.NS and WIPRO.NS. "
        "This balances the aggressive momentum stance but creates dependency on FMCG stability.\n\n"
        "## Risk & Concentration Warnings\n"
        "- **High concentration risk**: OBEROIRLTY.NS receives 18-19% of all positive attention—if this stock underperforms, the entire strategy suffers.\n"
        "- **Sector clustering in FMCG**: MARICO.NS and BRITANNIA.NS are tightly linked (4.2% attention weight), creating vulnerability to sector-wide shocks.\n"
        "- **Hedging is narrow**: 49% of negative attention centers on MARICO.NS alone, limiting downside protection if other sectors decline.\n"
    )

    payload = {
        "system_instruction": SYSTEM_PROMPT,
        "instructions": instructions,
        "example_output": example_block,
        "attention_summary": attn_summary,
    }

    return json.dumps(payload, indent=2)

def llm_generate(prompt: str, model="gemini-2.0-flash", retries=3, delay=5) -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY.")
    genai.configure(api_key=key)

    # Token limit check
    estimated_tokens = _estimate_tokens(prompt)
    if estimated_tokens > MAX_PROMPT_TOKENS:
        print(f"[WARN] Prompt has ~{estimated_tokens} tokens, truncating to ~{MAX_PROMPT_TOKENS}")
        prompt = _truncate_text(prompt, MAX_PROMPT_CHARS)

    llm = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_PROMPT)
    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Generating explanation with Gemini ({attempt}/{retries})...")
            resp = llm.generate_content(prompt, generation_config={"temperature": 0.3, "top_p": 0.9})
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
            raise RuntimeError("Empty response from LLM.")
        except Exception as e:
            msg = str(e)
            if "429" in msg and attempt < retries:
                print(f"[WARN] Rate limited, retrying in {delay}s...")
                time.sleep(delay)
                continue
            # Handle token limit errors
            if "token" in msg.lower() and "limit" in msg.lower() and attempt < retries:
                print(f"[WARN] Token limit exceeded, reducing prompt and retrying...")
                prompt = _truncate_text(prompt, MAX_PROMPT_CHARS // 2)
                continue
            print(f"[ERROR] Gemini call failed: {e}")
            return f"**LLM generation failed:** {e}"
    return "**LLM unavailable.**"

def main():
    args = parse_args()
    attention_path = Path(args.attention_json).expanduser()

    try:
        attn_summary = load_attention_summary(attention_path)
        print("[INFO] Loaded attention summary successfully.")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    prompt = assemble_prompt(attn_summary)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "hgat_attention_prompt.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"[INFO] Prompt saved at {prompt_path}")
    if args.llm:
        narrative = llm_generate(prompt, model=args.llm_model)
    else:
        narrative = f"Loaded HGAT attention summary for model: {attn_summary.get('model_path', 'N/A')}"
    out_path = Path(args.output)
    out_path.write_text(narrative, encoding="utf-8")
    print(f"[INFO] Narrative saved to {out_path}")

    if args.print:
        print("\n--- HGAT Attention Narrative ---\n")
        print(narrative)

    print(" Done.")

if __name__ == "__main__":
    main()