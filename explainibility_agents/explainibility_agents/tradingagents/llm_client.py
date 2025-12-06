# ======================================================================
# 🔹 ADDITIONAL CLIENT FUNCTIONS FOR TRADING AGENTS
# ======================================================================

import os
import sys
from typing import List, Optional

from .llm_provider import ProviderError, generate_completion

LAST_LLM_ERROR: Optional[str] = None  # for CLI error reporting
DEFAULT_MODEL = os.environ.get("TRADINGAGENTS_LLM_MODEL", "gpt-4.1-mini")
DEFAULT_PROVIDER = os.environ.get("TRADINGAGENTS_LLM_PROVIDER")


def _call_llm(
    prompt: str,
    *,
    system_prompt: str,
    model: Optional[str],
    max_points: int,
    temperature: float,
) -> List[str]:
    global LAST_LLM_ERROR
    try:
        text = generate_completion(
            prompt,
            system_prompt=system_prompt,
            model=model or DEFAULT_MODEL,
            temperature=temperature,
            top_p=0.9,
            provider=DEFAULT_PROVIDER,
        )
        points = [line.strip(" -*•") for line in text.splitlines() if line.strip()]
        return points[:max_points]
    except ProviderError as exc:
        LAST_LLM_ERROR = str(exc)
        print(f"[WARN] LLM generation failed: {exc}", file=sys.stderr)
        return []


def summarise_fundamentals(
    *,
    ticker: str,
    weight: float,
    as_of: str,
    metrics_table: str,
    metrics_summary: str,
    max_points: int = 4,
    model: Optional[str] = None,
) -> List[str]:
    """Summarise fundamental metrics using the configured LLM provider."""

    prompt = (
        f"You are an equity analyst creating a short fundamental summary for {ticker}.\n"
        f"Portfolio Weight: {weight:.2%}\nAs of: {as_of}\n\n"
        f"Metrics Table:\n{metrics_table}\n\nSummary of Metrics:\n{metrics_summary}\n\n"
        f"Write up to {max_points} key bullet points summarizing financial health, valuation, "
        f"and growth outlook in crisp language. Keep each bullet under 120 characters and do not offer services or monitoring."
    )
    return _call_llm(
        prompt,
        system_prompt="Summarise fundamentals for portfolio reporting with short observational bullets only (no offers or action items).",
        model=model,
        max_points=max_points,
        temperature=0.2,
    )


def summarise_news(
    *,
    ticker: str,
    weight: float,
    as_of: str,
    lookback_days: int,
    article_summaries: str,
    net_sentiment: int,
    max_points: int = 4,
    model: Optional[str] = None,
) -> List[str]:
    """Summarise news sentiment and coverage."""

    prompt = (
        f"You are a financial news analyst reviewing {ticker}.\n"
        f"Portfolio Weight: {weight:.2%}\nDate: {as_of}\nLookback: {lookback_days} days\n"
        f"Net Sentiment Score: {net_sentiment}\n\n"
        f"Recent Headlines:\n{article_summaries}\n\n"
        f"Summarize up to {max_points} insights capturing sentiment trends, risks, "
        f"and how news tone might affect allocation decisions."
        " Keep each bullet under 120 characters and do not offer additional services or monitoring."
    )
    return _call_llm(
        prompt,
        system_prompt="Summarise financial news coverage in short bullets without promising actions or monitoring.",
        model=model,
        max_points=max_points,
        temperature=0.1,
    )


def summarise_weight_points(
    *,
    ticker: str,
    weight: float,
    as_of: str,
    fundamental_points: List[str],
    news_points: List[str],
    metrics_table: str,
    news_table: str,
    max_points: int = 6,
    model: Optional[str] = None,
) -> List[str]:
    """Combine fundamental + news rationales into unified LLM summary."""

    prompt = (
        f"As an investment strategist, create {max_points} concise, bullet-point insights "
        f"that integrate fundamental and news perspectives for {ticker}.\n\n"
        f"Portfolio Weight: {weight:.2%} (as of {as_of})\n\n"
        f"Fundamental Rationale Points:\n"
        + "\n".join(f"- {pt}" for pt in fundamental_points)
        + "\n\nNews-Based Points:\n"
        + "\n".join(f"- {pt}" for pt in news_points)
        + "\n\nFundamental Metrics:\n"
        + metrics_table
        + "\n\nRecent News Table:\n"
        + news_table
        + "\n\nWrite objective, data-grounded bullet points combining both analyses, "
        f"and avoid repetition. Do not offer to perform monitoring, run scenarios, or describe future services. Keep each bullet under 120 characters."
    )

    return _call_llm(
        prompt,
        system_prompt="Merge fundamental and news rationales into concise observational bullets (no offers or action items).",
        model=model,
        max_points=max_points,
        temperature=0.3,
    )
