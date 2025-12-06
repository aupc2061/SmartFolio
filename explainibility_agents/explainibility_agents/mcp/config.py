from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class XAIRequest:
    """Normalized configuration payload for MCP-triggered XAI runs."""

    date: str
    monthly_log_csv: str
    model_path: str
    market: str = "hs300"
    data_root: str = "dataset_default"
    top_k: int = 5
    lookback_days: int = 60
    llm: bool = False
    llm_model: str = "gpt-4.1-mini"
    output_dir: str | Path = "explainability_results"
    monthly_run_id: Optional[str] = None
    latent: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser()
        self.monthly_log_csv = str(Path(self.monthly_log_csv).expanduser())
        self.model_path = str(Path(self.model_path).expanduser())
        self.data_root = str(Path(self.data_root).expanduser())

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "XAIRequest":
        # Ensure payload is a dict
        if not isinstance(payload, dict):
            try:
                payload = {k: payload[k] for k in payload}
            except Exception:
                # If conversion fails, we can't do much, but let's try to proceed
                pass

        candidate = payload.get("config") if isinstance(payload, dict) else None
        
        # Use the payload directly if it's a dict, otherwise try to convert
        source = candidate or payload
        if isinstance(source, dict):
            data = dict(source)
        else:
            # Fallback for non-dict objects that might be iterable (like Pathway Json)
            # This avoids dict(source) which fails if source iterates keys
            try:
                data = {k: source[k] for k in source}
            except Exception:
                 # Last resort: try dict() and hope for the best
                data = dict(source)

        extra = data.pop("extra", {})
        obj = cls(**data)
        obj.extra.update(extra)
        return obj

    def rolling_window(self) -> Tuple[str, str]:
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        end_date = _dt.datetime.strptime(self.date, "%Y-%m-%d").date()
        start_date = end_date - _dt.timedelta(days=self.lookback_days - 1)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    def to_orchestrator_config(self):  # lazy import to avoid circular refs
        from explainibility_agents.orchestrator_xai import OrchestratorConfig

        return OrchestratorConfig(
            date=self.date,
            monthly_log_csv=Path(self.monthly_log_csv),
            model_path=self.model_path,
            market=self.market,
            data_root=self.data_root,
            top_k=int(self.top_k),
            lookback_days=int(self.lookback_days),
            llm=bool(self.llm),
            llm_model=self.llm_model,
            output_dir=self.output_dir,
            monthly_run_id=self.monthly_run_id,
            latent=bool(self.latent),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "monthly_log_csv": self.monthly_log_csv,
            "model_path": self.model_path,
            "market": self.market,
            "data_root": self.data_root,
            "top_k": self.top_k,
            "lookback_days": self.lookback_days,
            "llm": self.llm,
            "llm_model": self.llm_model,
            "output_dir": str(self.output_dir),
            "monthly_run_id": self.monthly_run_id,
            "latent": self.latent,
            "extra": self.extra,
        }
