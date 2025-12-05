"""
Shared state management for the streaming pipeline.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd

from streaming.config import (
    NUM_RISK_GROUPS,
    MONTHLY_STOCK_DATA_DIR,
    USER_DATABASE_DIR,
    get_risk_group,
)


@dataclass
class StockRow:
    """A single row of stock data."""
    date: str
    ticker: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float


class StockBuffer:
    """
    In-memory buffer for stock data.
    
    Stores stock rows until a month change is detected,
    then flushes to a CSV file.
    """
    
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []
        self.current_month: Optional[str] = None
        self.tickers: List[str] = []
        self._buffer: List[Dict[str, Any]] = []  # Alias for rows
        
    def add_row(self, date: str, stock_data: Dict[str, Dict[str, float]]) -> Tuple[bool, Optional[str]]:
        """
        Add a row of stock data.
        
        Args:
            date: Date string (YYYY-MM-DD)
            stock_data: Dict mapping ticker -> {open, high, low, close, adj_close, volume}
            
        Returns:
            Tuple of (month_changed, saved_path_or_None)
        """
        month_changed = False
        saved_path = None
        
        # Check for month change
        new_month = self.get_current_month(date)
        if self.current_month is not None and new_month != self.current_month:
            # Flush the old month's data
            saved_path = self.flush_to_csv()
            month_changed = True
        
        # Update current month
        self.current_month = new_month
        
        # Add the data
        for ticker, ohlcv in stock_data.items():
            row = {
                "date": date,
                "kdcode": ticker,
                "dt": date,
                "open": ohlcv.get("open", ohlcv.get("Open", 0)),
                "high": ohlcv.get("high", ohlcv.get("High", 0)),
                "low": ohlcv.get("low", ohlcv.get("Low", 0)),
                "close": ohlcv.get("close", ohlcv.get("Close", 0)),
                "adj_close": ohlcv.get("adj_close", ohlcv.get("Adj Close", 0)),
                "volume": ohlcv.get("volume", ohlcv.get("Volume", 0)),
            }
            self.rows.append(row)
            self._buffer.append(row)
            if ticker not in self.tickers:
                self.tickers.append(ticker)
        
        return month_changed, str(saved_path) if saved_path else None
    
    def get_current_month(self, date: str) -> str:
        """Extract month (YYYY-MM) from date string."""
        return date[:7]
    
    def check_month_change(self, new_date: str) -> bool:
        """
        Check if the new date represents a month change.
        
        Returns:
            True if month changed, False otherwise
        """
        if not self.rows:
            return False
        
        new_month = self.get_current_month(new_date)
        if self.current_month is None:
            self.current_month = new_month
            return False
        
        return new_month != self.current_month
    
    def flush_to_csv(self) -> Optional[Path]:
        """
        Flush current buffer to a CSV file.
        
        Returns:
            Path to the saved CSV file, or None if buffer is empty
        """
        if not self.rows:
            return None
        
        month = self.current_month
        if month is None and self.rows:
            month = self.get_current_month(self.rows[0]["date"])
        
        df = pd.DataFrame(self.rows)
        
        # Ensure directory exists
        MONTHLY_STOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save to monthly CSV
        csv_path = MONTHLY_STOCK_DATA_DIR / f"{month}.csv"
        df.to_csv(csv_path, index=False)
        
        # Clear buffer
        self.rows = []
        self._buffer = []
        
        return csv_path
    
    def update_current_month(self, date: str):
        """Update the current month tracker."""
        self.current_month = self.get_current_month(date)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert buffer to DataFrame."""
        return pd.DataFrame(self.rows)


class SharedState:
    """
    Centralized shared state for the streaming pipeline.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize shared state."""
        # Stock data buffer
        self.stock_buffer = StockBuffer()
        
        # Fine-tuning counters (one per risk group)
        self.finetune_counters: Dict[int, int] = {
            i: 0 for i in range(NUM_RISK_GROUPS)
        }
        
        # User risk assignments: user_id -> risk_group
        self.user_risk_assignments: Dict[str, int] = {}
        
        # User database records
        self.user_records: List[Dict[str, Any]] = []
        
        # Processing statistics
        self.stats = {
            "stock_rows_processed": 0,
            "users_processed": 0,
            "months_flushed": 0,
            "finetunes_completed": {i: 0 for i in range(NUM_RISK_GROUPS)},
            "start_time": datetime.now().isoformat(),
        }
    
    def add_stock_row(self, date: str, stock_data: Dict[str, Dict[str, float]]) -> bool:
        """
        Add a stock data row.
        
        Returns:
            True if a month change was detected (buffer flushed), False otherwise
        """
        month_changed = self.stock_buffer.check_month_change(date)
        
        if month_changed:
            # Flush the old month's data first
            self.stock_buffer.flush_to_csv()
            self.stats["months_flushed"] += 1
        
        # Add new row and update month
        self.stock_buffer.add_row(date, stock_data)
        self.stock_buffer.update_current_month(date)
        self.stats["stock_rows_processed"] += 1
        
        return month_changed
    
    def add_user(self, user_data: Dict[str, Any], risk_score: float, risk_group: int):
        """Add a processed user record."""
        record = {
            **user_data,
            "risk_score": risk_score,
            "risk_group": risk_group,
            "processed_at": datetime.now().isoformat(),
        }
        self.user_records.append(record)
        self.user_risk_assignments[str(user_data.get("userid", user_data.get("id", "")))] = risk_group
        self.stats["users_processed"] += 1
        
        # Persist to file
        self._persist_user(record)
    
    def _persist_user(self, record: Dict[str, Any]):
        """Persist a user record to JSONL file."""
        filepath = USER_DATABASE_DIR / "users.jsonl"
        with open(filepath, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def get_finetune_counter(self, risk_group: int) -> int:
        """Get the finetune counter for a risk group."""
        return self.finetune_counters.get(risk_group, 0)
    
    def reset_finetune_counter(self, risk_group: int) -> int:
        """
        Reset and return the finetune counter for a risk group.
        
        Returns:
            The value before reset
        """
        value = self.finetune_counters.get(risk_group, 0)
        self.finetune_counters[risk_group] = 0
        return value
    
    def increment_all_finetune_counters(self):
        """Increment all finetune counters (called on month change)."""
        for i in range(NUM_RISK_GROUPS):
            self.finetune_counters[i] += 1
    
    def record_finetune_complete(self, risk_group: int):
        """Record that a finetune was completed for a risk group."""
        self.stats["finetunes_completed"][risk_group] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            "current_time": datetime.now().isoformat(),
            "buffer_size": len(self.stock_buffer.rows),
            "current_month": self.stock_buffer.current_month,
        }
    
    def get_user_model(self, user_id: str) -> Optional[int]:
        """Get the model (risk group) assigned to a user."""
        return self.user_risk_assignments.get(user_id)
    
    def add_user_risk_data(self, userid: int, score: float, label: str, group: int):
        """Add user risk data (called by user consumer)."""
        record = {
            "userid": userid,
            "risk_score": score,
            "risk_label": label,
            "risk_group": group,
            "processed_at": datetime.now().isoformat(),
        }
        self.user_records.append(record)
        self.user_risk_assignments[str(userid)] = group
        self.stats["users_processed"] += 1
        
        # Persist to file
        self._persist_user(record)
    
    def get_all_user_risk_data(self) -> List[Dict[str, Any]]:
        """Get all user risk data records."""
        return self.user_records
    
    def increment_finetune_counter(self, risk_group: int):
        """Increment the finetune counter for a risk group."""
        if risk_group in self.finetune_counters:
            self.finetune_counters[risk_group] += 1


# Global singleton instance
shared_state = SharedState()
