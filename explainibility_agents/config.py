"""Configuration objects for the LangChain-powered explainibility pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Holding:
    ticker: str
    weight: float
    as_of: Optional[str]


@dataclass
class ExplainabilityConfig:
    date: str
    monthly_log_csv: Path
    model_path: str
    market: str = "custom"
    data_root: str = "dataset_default"
    top_k: int = 5
    lookback_days: int = 30
    llm: bool = False
    llm_model: str = "gpt-4.1-mini"
    output_dir: Path = Path("explainability_results")
    monthly_run_id: Optional[str] = None
    latent: bool = False

    def ensure_output_dir(self) -> Path:
        path = self.output_dir.expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
