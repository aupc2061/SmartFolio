"""
Utility functions for the streaming pipeline.
"""

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Import SMARTFOLIO_DIR from config for path resolution
try:
    from streaming.config import SMARTFOLIO_DIR
except ImportError:
    # Fallback: detect SmartFolio root
    _current = Path(__file__).resolve().parent.parent  # streaming/
    SMARTFOLIO_DIR = _current.parent  # SmartFolio/


def _resolve_data_path(path_str: str) -> str:
    """
    Resolve a data path that may be relative or use /data/ prefix.
    
    Args:
        path_str: Path string (e.g., '/data/images/test.png' or 'data/images/test.png')
        
    Returns:
        Resolved absolute path
    """
    if not path_str:
        return path_str
        
    # Handle relative paths that start with /data/
    if path_str.startswith("/data/"):
        # In Docker, files are at /app/data/, locally at SmartFolio/data/
        docker_path = "/app" + path_str  # /app/data/...
        local_path = str(SMARTFOLIO_DIR / path_str.lstrip("/"))
        
        if os.path.exists(docker_path):
            return docker_path
        elif os.path.exists(local_path):
            return local_path
        # Return docker path as default (for Docker environment)
        return docker_path
    
    return path_str


def encode_image_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    image_path = _resolve_data_path(image_path)
    
    if not os.path.exists(image_path):
        # Return empty string if file doesn't exist
        return ""
    
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_video_base64(video_path: str) -> str:
    """
    Encode a video file to base64 string.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Base64 encoded string
    """
    video_path = _resolve_data_path(video_path)
    
    if not os.path.exists(video_path):
        # Return empty string if file doesn't exist
        return ""
    
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_image_base64(base64_string: str) -> bytes:
    """
    Decode a base64 string to image bytes.
    
    Args:
        base64_string: Base64 encoded string
        
    Returns:
        Image bytes
    """
    if not base64_string:
        return b""
    return base64.b64decode(base64_string)


def decode_video_base64(base64_string: str) -> bytes:
    """
    Decode a base64 string to video bytes.
    
    Args:
        base64_string: Base64 encoded string
        
    Returns:
        Video bytes
    """
    if not base64_string:
        return b""
    return base64.b64decode(base64_string)


def parse_stock_row(date: str, data_str: str) -> Dict[str, Dict[str, float]]:
    """
    Parse a stock data row from the serialized format.
    
    Args:
        date: Date string (YYYY-MM-DD)
        data_str: Serialized stock data in format:
                  "TICKER|open|high|low|close|adj_close|volume;TICKER2|..."
    
    Returns:
        Dict mapping ticker -> {open, high, low, close, adj_close, volume}
    """
    result = {}
    
    if not data_str:
        return result
    
    # Split by semicolon for each ticker
    ticker_parts = data_str.split(";")
    
    for part in ticker_parts:
        if not part.strip():
            continue
        
        fields = part.split("|")
        if len(fields) >= 7:
            ticker = fields[0]
            try:
                result[ticker] = {
                    "open": float(fields[1]) if fields[1] else 0.0,
                    "high": float(fields[2]) if fields[2] else 0.0,
                    "low": float(fields[3]) if fields[3] else 0.0,
                    "close": float(fields[4]) if fields[4] else 0.0,
                    "adj_close": float(fields[5]) if fields[5] else 0.0,
                    "volume": float(fields[6]) if fields[6] else 0.0,
                }
            except (ValueError, IndexError):
                continue
    
    return result


def format_stock_row(date: str, stock_data: pd.Series, tickers: List[str]) -> str:
    """
    Format a stock data row for Kafka transmission.
    
    Args:
        date: Date string
        stock_data: Series from ohlcv_raw.csv with MultiIndex columns
        tickers: List of tickers
        
    Returns:
        Serialized string in format:
        "TICKER|open|high|low|close|adj_close|volume;TICKER2|..."
    """
    parts = []
    
    for ticker in tickers:
        try:
            open_val = stock_data.get((ticker, "Open"), 0)
            high_val = stock_data.get((ticker, "High"), 0)
            low_val = stock_data.get((ticker, "Low"), 0)
            close_val = stock_data.get((ticker, "Close"), 0)
            adj_close_val = stock_data.get((ticker, "Adj Close"), 0)
            volume_val = stock_data.get((ticker, "Volume"), 0)
            
            # Handle NaN values
            def safe_float(val):
                if pd.isna(val):
                    return ""
                return str(val)
            
            part = f"{ticker}|{safe_float(open_val)}|{safe_float(high_val)}|{safe_float(low_val)}|{safe_float(close_val)}|{safe_float(adj_close_val)}|{safe_float(volume_val)}"
            parts.append(part)
        except Exception:
            continue
    
    return ";".join(parts)


def load_ohlcv_dataframe(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the ohlcv_raw.csv file with MultiIndex columns.
    
    Args:
        csv_path: Path to ohlcv_raw.csv (can be str or Path)
        
    Returns:
        Tuple of (DataFrame, list of tickers)
    """
    # Read with multi-level headers (first two rows)
    df = pd.read_csv(str(csv_path), header=[0, 1], index_col=0)
    
    # Extract unique tickers from the column names
    tickers = list(set([col[0] for col in df.columns if col[0] != "Ticker"]))
    
    return df, tickers


def load_user_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load the input.csv user data file.
    
    Args:
        csv_path: Path to input.csv
        
    Returns:
        DataFrame with user data
    """
    return pd.read_csv(csv_path)


def prepare_user_message(row: pd.Series, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Prepare a user data message for Kafka.
    
    Args:
        row: A row from the user DataFrame (new format with document paths)
        base_dir: Optional base directory for resolving document paths
        
    Returns:
        Dict suitable for JSON serialization
    """
    # Helper function to resolve paths
    def resolve_path(path_str: str) -> str:
        if not path_str:
            return ""
        path_str = str(path_str)
        # Use the centralized path resolver
        resolved = _resolve_data_path(path_str)
        if resolved != path_str:
            return resolved
        # Handle base_dir for relative paths
        if base_dir and not os.path.isabs(path_str):
            return str(base_dir / path_str)
        return path_str
    
    # Get document paths
    aadhar_path = resolve_path(row.get("aadhar_path", ""))
    pan_path = resolve_path(row.get("pan_path", ""))
    itr_path = resolve_path(row.get("itr_path", ""))
    video_path = resolve_path(row.get("video_path", ""))
    
    # print(aadhar_path, pan_path, itr_path, video_path)
    
    
    # Encode image documents to base64 (small files)
    aadhar_base64 = encode_image_base64(aadhar_path) if aadhar_path else ""
    pan_base64 = encode_image_base64(pan_path) if pan_path else ""
    itr_base64 = encode_image_base64(itr_path) if itr_path else ""
    # Video: send path only (too large for Kafka), consumer reads from shared volume
    # video_base64 is intentionally empty - consumer will use video_path
    
    
    # print(aadhar_base64, pan_base64, itr_base64, video_base64)
    return {
        "userid": int(row.get("id", 0)),
        "first": str(row.get("first", "")),
        "last": str(row.get("last", "")),
        # Document paths (for reference)
        "aadhar_path": str(row.get("aadhar_path", "")),
        "pan_path": str(row.get("pan_path", "")),
        "itr_path": str(row.get("itr_path", "")),
        "video_path": str(row.get("video_path", "")),
        # Base64 encoded documents (images only)
        "aadhar_base64": aadhar_base64,
        "pan_base64": pan_base64,
        "itr_base64": itr_base64,
        "video_base64": "",  # Video too large - consumer reads from video_path directly
        # Additional details from CSV
        "main_occupation": str(row.get("main_occupation", "")),
        "marital_status": str(row.get("marital_status", "")),
        "dependents": int(row.get("dependents", 0)) if pd.notna(row.get("dependents")) else 0,
        # Questionnaire
        "Q1": str(row.get("Q1", "")),
        "Q2": str(row.get("Q2", "")),
        "Q3": str(row.get("Q3", "")),
        "Q4": str(row.get("Q4", "")),
        "Q5": str(row.get("Q5", "")),
        "Q6": str(row.get("Q6", "")),
    }


def get_monthly_csv_files(directory: str) -> List[Tuple[str, Path]]:
    """
    Get all monthly CSV files sorted by month.
    
    Args:
        directory: Path to the monthly stock data directory
        
    Returns:
        List of (month_string, file_path) tuples sorted by month
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    files = []
    for f in dir_path.glob("*.csv"):
        # Extract month from filename (expected format: YYYY-MM.csv)
        month = f.stem  # e.g., "2024-11"
        files.append((month, f))
    
    return sorted(files, key=lambda x: x[0])
