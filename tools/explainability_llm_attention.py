#!/usr/bin/env python
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

# -----------------------------
# CONSTANTS
# -----------------------------
SYSTEM_PROMPT = (
    "You are a quantitative explainability analyst specializing in graph-based models. "
    "You will analyze the HGAT (Hierarchical Graph Attention Network) explainability summary "
    "used within a reinforcement-learning portfolio policy. "
    "Your goal is to interpret semantic attention weights, inter-stock relationships, "
    "and cross-stock influence patterns clearly and concisely for a technical audience."
)


# -----------------------------
# ARGUMENTS
# -----------------------------
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


# -----------------------------
# LOAD ATTENTION SUMMARY
# -----------------------------
def load_attention_summary(attention_path: Path):
    if not attention_path.exists():
        raise FileNotFoundError(f"Attention summary file not found: {attention_path}")

    with open(attention_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # if file already contains 'attention_summary' at root
    if "attention_summary" in data:
        attn = data["attention_summary"]
    else:
        attn = data

    # compress large arrays for readability
    summary = {
        "model_path": attn.get("model_path", ""),
        "market": attn.get("market", ""),
        "num_stocks": attn.get("num_stocks", ""),
        "semantic_labels": attn.get("semantic_labels", []),
        "semantic_mean": attn.get("semantic_mean", []),
        "top_edges": attn.get("top_edges", {}),
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


# -----------------------------
# PROMPT ASSEMBLY
# -----------------------------
def assemble_prompt(attn_summary: dict) -> str:
    """
    Builds a precise, LLM-ready prompt for HGAT interpretability.
    """
    instructions = (
        "### Task\n"
        "Analyze this HGAT attention summary for a reinforcement-learning portfolio allocator.\n\n"
        "Explain the following aspects:\n"
        "1. Semantic attention distribution — interpret each label (Self, Industry, Positive, Negative) "
        "and what their mean values imply about attention focus.\n"
        "2. Positive and negative attention edges — explain the relationships and leading nodes.\n"
        "3. Industry edge patterns — highlight any notable clustering or dependency trends.\n"
        "4. Summarize portfolio-level implications such as diversification, sector dependencies, or hedging structures.\n\n"
        "### Output Format\n"
        "- Section 1: Overview (market, num_stocks, checkpoint)\n"
        "- Section 2: Semantic attention interpretation\n"
        "- Section 3: Edge-level attention insights (industry, positive, negative)\n"
        "- Section 4: Portfolio interpretation summary\n\n"
        "Keep the tone analytical and quant-oriented, avoid generic statements."
    )

    example_block = (
        "Example:\n"
        "The HGAT network allocates higher mean attention to Positive (≈0.43) and Industry (≈0.20), "
        "indicating strong inter-stock relational reasoning and mild sector correlation learning. "
        "Negative attention centered on MARICO.NS suggests a hedging structure balancing overexposure "
        "from OBEROIRLTY.NS (a positive influencer)."
    )

    payload = {
        "system_instruction": SYSTEM_PROMPT,
        "instructions": instructions,
        "example_output": example_block,
        "attention_summary": attn_summary,
    }

    return json.dumps(payload, indent=2)


# -----------------------------
# GEMINI GENERATION
# -----------------------------
def llm_generate(prompt: str, model="gemini-2.0-flash", retries=3, delay=5) -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY.")
    genai.configure(api_key=key)

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
            print(f"[ERROR] Gemini call failed: {e}")
            return f"**LLM generation failed:** {e}"
    return "**LLM unavailable.**"


# -----------------------------
# MAIN
# -----------------------------
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

    # Save prompt for reproducibility
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "hgat_attention_prompt.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"[INFO] Prompt saved at {prompt_path}")

    # Generate LLM explanation
    if args.llm:
        narrative = llm_generate(prompt, model=args.llm_model)
    else:
        narrative = f"Loaded HGAT attention summary for model: {attn_summary.get('model_path', 'N/A')}"

    # Save final narrative
    out_path = Path(args.output)
    out_path.write_text(narrative, encoding="utf-8")
    print(f"[INFO] Narrative saved to {out_path}")

    if args.print:
        print("\n--- HGAT Attention Narrative ---\n")
        print(narrative)
        print("\n--------------------------------")

    print(" Done.")


if __name__ == "__main__":
    main()
