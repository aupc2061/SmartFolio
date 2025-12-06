"""
Simple monthly dataset updater for SmartFolio.

This module fetches the latest month's data from yfinance and builds
daily pkl files in the same format as build_dataset_yf.py.

No manifest, no shards - just simple date-based pkl files.

Usage:
    python -m gen_data.update_monthly_dataset --market custom --tickers_file tickers.csv
    
Or call from fine_tune_month():
    from gen_data.update_monthly_dataset import fetch_latest_month_data
    fetch_latest_month_data(args)
"""

from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from pandas.tseries.offsets import MonthEnd
from torch.autograd import Variable
from torch_geometric.data import Data

try:
    from .build_dataset_yf import (
        DATASET_CORR_ROOT,
        DATASET_DEFAULT_ROOT,
        FEATURE_COLS,
        FEATURE_COLS_NORM,
        cal_rolling_mean_std,
        fetch_ohlcv_yf,
        fetch_ohlcv_streaming_csv,
        filter_code,
        gen_mats_by_threshold,
        get_label,
        group_and_norm,
        build_industry_matrix,
    )
except ImportError:
    from build_dataset_yf import (
        DATASET_CORR_ROOT,
        DATASET_DEFAULT_ROOT,
        FEATURE_COLS,
        FEATURE_COLS_NORM,
        cal_rolling_mean_std,
        fetch_ohlcv_yf,
        fetch_ohlcv_streaming_csv,
        filter_code,
        gen_mats_by_threshold,
        get_label,
        group_and_norm,
        build_industry_matrix,
    )


def _update_index_csv(df_raw: pd.DataFrame, market: str) -> None:
    """
    Update the index CSV with new daily returns for benchmark comparison.
    Computes equal-weighted average daily return across all tickers.
    """
    try:
        idx_dir = os.path.join(DATASET_DEFAULT_ROOT, "index_data")
        os.makedirs(idx_dir, exist_ok=True)
        idx_path = os.path.join(idx_dir, f"{market}_index.csv")
        
        # Compute daily return per ticker
        df_idx = df_raw.copy()
        if "prev_close" not in df_idx.columns:
            df_idx = df_idx.sort_values(["kdcode", "dt"])
            df_idx["prev_close"] = df_idx.groupby("kdcode")["close"].shift(1)
            df_idx = df_idx.dropna(subset=["prev_close"])
        
        df_idx["daily_return"] = df_idx["close"] / df_idx["prev_close"] - 1
        
        # Equal-weighted average across tickers per day
        new_index = df_idx.groupby("dt")["daily_return"].mean().reset_index()
        new_index = new_index.rename(columns={"dt": "datetime"})
        
        # Load existing index and append new rows (avoid duplicates)
        if os.path.exists(idx_path):
            existing = pd.read_csv(idx_path)
            existing_dates = set(existing["datetime"].tolist())
            new_rows = new_index[~new_index["datetime"].isin(existing_dates)]
            if not new_rows.empty:
                combined = pd.concat([existing, new_rows], ignore_index=True)
                combined = combined.sort_values("datetime").reset_index(drop=True)
                combined.to_csv(idx_path, index=False)
                print(f"Updated index CSV with {len(new_rows)} new dates")
            else:
                print("Index CSV already has all dates, no update needed")
        else:
            new_index.to_csv(idx_path, index=False)
            print(f"Created index CSV with {len(new_index)} dates")
    except Exception as e:
        print(f"Warning: Failed to update index CSV: {e}")


def get_existing_dates(data_dir: str) -> List[datetime]:
    """Scan pkl files in data_dir and return list of dates."""
    if not os.path.exists(data_dir):
        return []
    
    dates = []
    for f in os.listdir(data_dir):
        if f.endswith('.pkl'):
            try:
                date_str = f.replace('.pkl', '')
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(dt)
            except ValueError:
                continue
    return sorted(dates)


def get_next_month_range(latest_date: datetime) -> tuple:
    """Given the latest date, return start and end of the NEXT month."""
    # Move to first day of next month
    if latest_date.month == 12:
        next_month_start = datetime(latest_date.year + 1, 1, 1)
    else:
        next_month_start = datetime(latest_date.year, latest_date.month + 1, 1)
    
    # End of that month
    next_month_end = (next_month_start + MonthEnd(1)).to_pydatetime()
    
    # Don't fetch future dates
    today = datetime.now()
    if next_month_end > today:
        next_month_end = today
    
    return next_month_start, next_month_end


def save_daily_pkl(
    dt: str,
    df_all: pd.DataFrame,
    codes: List[str],
    market: str,
    horizon: int,
    relation_type: str,
    lookback: int = 20,
    threshold: float = 0.5,
    norm: bool = True,
    industry_mat: Optional[np.ndarray] = None,
):
    """Save a single day's pkl file."""
    stock_trade_dt_s_all = sorted(df_all["dt"].unique().tolist())
    
    if dt not in stock_trade_dt_s_all:
        return False
    
    dt_idx = stock_trade_dt_s_all.index(dt)
    if dt_idx < lookback - 1:
        # Not enough history for this date
        return False
    
    # Time series window
    ts_start = stock_trade_dt_s_all[dt_idx - (lookback - 1)]
    df_ts = df_all[(df_all["dt"] >= ts_start) & (df_all["dt"] <= dt)].copy()
    
    # Industry matrix
    if industry_mat is None:
        industry_mat = np.eye(len(codes), dtype=np.float32)
    ind = torch.from_numpy(industry_mat.astype(np.float32))
    
    # Compute correlation for this month
    month_tag = dt[:7]  # YYYY-MM
    month_days = [d for d in stock_trade_dt_s_all if d.startswith(month_tag) and d <= dt]
    
    # Use available data for correlation
    if len(month_days) >= 2:
        end_date = month_days[-1]
        end_idx = stock_trade_dt_s_all.index(end_date)
        start_idx = max(0, end_idx - (lookback - 1))
        start_date = stock_trade_dt_s_all[start_idx]
        window = df_all[(df_all["dt"] >= start_date) & (df_all["dt"] <= end_date)]
        
        feat_dict = {}
        actual_lookback = end_idx - start_idx + 1
        for code in codes:
            sub = window[window["kdcode"] == code]
            y = sub[["close", "open", "high", "low", "prev_close", "volume"]].values
            if y.shape[0] == actual_lookback and y.shape[0] >= 2:
                feat_dict[code] = y.reshape(-1)
        
        if len(feat_dict) >= 2:
            valid_codes = [c for c in codes if c in feat_dict]
            X = np.stack([feat_dict[c] for c in valid_codes], axis=0)
            X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
            corr_mat = np.corrcoef(X)
            corr_df = pd.DataFrame(corr_mat, index=valid_codes, columns=valid_codes).fillna(0)
            for i in range(len(valid_codes)):
                corr_df.iat[i, i] = 1.0
            # Reindex to match codes order
            corr_df = corr_df.reindex(index=codes, columns=codes).fillna(0)
        else:
            corr_df = pd.DataFrame(np.eye(len(codes)), index=codes, columns=codes)
    else:
        corr_df = pd.DataFrame(np.eye(len(codes)), index=codes, columns=codes)
    
    pos_adj, neg_adj = gen_mats_by_threshold(corr_df, threshold)
    corr = torch.from_numpy(corr_df.values.astype(np.float32))
    pos = torch.from_numpy(pos_adj.astype(np.float32))
    neg = torch.from_numpy(neg_adj.astype(np.float32))
    
    # Build features
    ts_features = []
    features = []
    labels = []
    valid_codes_for_sample = []
    
    cols = FEATURE_COLS_NORM if norm else FEATURE_COLS
    skipped_reasons = {"no_ts_data": 0, "ts_shape_mismatch": 0, "no_dt_data": 0}
    for code in codes:
        df_ts_code = df_ts[df_ts["kdcode"] == code]
        ts_array = df_ts_code[cols].values
        df_code_dt = df_ts_code[df_ts_code["dt"] == dt]
        array = df_code_dt[cols].values
        
        if ts_array.shape[0] == 0:
            skipped_reasons["no_ts_data"] += 1
            continue
        if ts_array.shape[0] != lookback:
            skipped_reasons["ts_shape_mismatch"] += 1
            continue
        if array.shape[0] != 1:
            skipped_reasons["no_dt_data"] += 1
            continue
            
        ts_features.append(ts_array)
        features.append(array[0])
        valid_codes_for_sample.append(code)
        label = df_ts_code.loc[df_ts_code["dt"] == dt]["label"].values
        labels.append(label[0] if len(label) > 0 else 0.0)
    
    if not ts_features:
        total_skipped = sum(skipped_reasons.values())
        print(f"  Skipping {dt}: no valid features. Reasons: {skipped_reasons} (total codes: {len(codes)})")
        return False
    
    ts_features = torch.from_numpy(np.array(ts_features)).float()
    features = torch.from_numpy(np.array(features)).float()
    labels = torch.tensor(labels, dtype=torch.float32)
    
    # PyG data
    edge_index = torch.triu_indices(ind.size(0), ind.size(0), offset=1)
    pyg_data = Data(x=features, edge_index=edge_index)
    pyg_data.edge_attr = ind[edge_index[0], edge_index[1]]
    
    result = {
        "corr": Variable(corr),
        "ts_features": Variable(ts_features),
        "features": Variable(features),
        "industry_matrix": Variable(ind),
        "pos_matrix": Variable(pos),
        "neg_matrix": Variable(neg),
        "pyg_data": pyg_data,
        "labels": Variable(labels),
        "mask": [True] * len(labels),
    }
    
    # Sanitize NaNs
    for k, v in list(result.items()):
        if isinstance(v, torch.Tensor):
            result[k] = torch.nan_to_num(v, nan=0.0)
    
    # Save
    save_dir = os.path.join(DATASET_DEFAULT_ROOT, f"data_train_predict_{market}", f"{horizon}_{relation_type}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{dt}.pkl"), "wb") as f:
        pickle.dump(result, f)
    
    return True


def fetch_latest_month_data(
    market: str = "custom",
    horizon: int = 1,
    relation_type: str = "hy",
    tickers_file: str = "tickers.csv",
    lookback: int = 20,
    threshold: float = 0.5,
    norm: bool = True,
    stream: Optional[object] = None,
) -> str:
    """
    Fetch the latest month's data from yfinance and build pkl files.
    
    Returns the month label (YYYY-MM) that was fetched.
    """
    # Ensure horizon is int (may come as string from CLI args)
    horizon = int(horizon)
    
    # Data directory
    data_dir = os.path.join(DATASET_DEFAULT_ROOT, f"data_train_predict_{market}", f"{horizon}_{relation_type}")
    
    # Get existing dates
    existing_dates = get_existing_dates(data_dir)
    
    if not existing_dates:
        raise ValueError(f"No existing pkl files found in {data_dir}. Run build_dataset_yf.py first.")
    
    latest_date = max(existing_dates)
    print(f"Latest existing date: {latest_date.strftime('%Y-%m-%d')}")
    
    # Get next month range
    next_start, next_end = get_next_month_range(latest_date)
    
    # Check if we're trying to fetch future data
    today = datetime.now()
    if next_start > today:
        raise ValueError(f"Next month ({next_start.strftime('%Y-%m')}) is in the future. No data to fetch.")
    
    print(f"Fetching data for: {next_start.strftime('%Y-%m-%d')} to {next_end.strftime('%Y-%m-%d')}")
    
    # Load tickers
    if os.path.exists(tickers_file):
        df_t = pd.read_csv(tickers_file)
        col = "kdcode" if "kdcode" in df_t.columns else ("ticker" if "ticker" in df_t.columns else None)
        if col is None:
            raise ValueError("tickers_file must have a 'kdcode' or 'ticker' column")
        tickers = sorted(df_t[col].dropna().astype(str).unique().tolist())
    else:
        raise FileNotFoundError(f"Tickers file not found: {tickers_file}")
    
    # We need extra history for lookback, so start earlier
    # Use 3x lookback in calendar days to account for weekends/holidays
    fetch_start = (next_start - timedelta(days=lookback * 3)).strftime("%Y-%m-%d")
    fetch_end = next_end.strftime("%Y-%m-%d")
    
    # Fetch from yfinance
    print(f"Downloading OHLCV for {len(tickers)} tickers...")
    if stream is not None:
        # Use the streaming output directory where StockBuffer.flush_to_csv() saves files
        month_tag = (next_start + timedelta(days=2)).strftime('%Y-%m')
        # Import streaming config for the correct path
        try:
            from streaming.config import MONTHLY_STOCK_DATA_DIR
            month_csv_path = str(MONTHLY_STOCK_DATA_DIR / f"{month_tag}.csv")
        except ImportError:
            # Fallback to relative path from SmartFolio root
            month_csv_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "streaming", "output", "monthly_stock_data", f"{month_tag}.csv"
            )
        df_raw = fetch_ohlcv_streaming_csv(tickers, month_csv_path=month_csv_path, lock=stream)
    else:
        df_raw = fetch_ohlcv_yf(tickers, fetch_start, fetch_end)
    
    if df_raw.empty:
        raise ValueError("No data returned from yfinance")
    
    # Process data
    # NOTE: get_label() drops last `horizon` dates (no forward return)
    df_lbl = get_label(df_raw, horizon=horizon)
    df_roll = cal_rolling_mean_std(df_lbl, cal_cols=["close", "volume"], lookback=5, use_pathway=False)
    df_norm = group_and_norm(df_roll, base_cols=["close_mean", "close_std", "volume_mean", "volume_std"], n_clusters=4)
    
    # Update the index CSV AFTER get_label so dates match pkl files
    _update_index_csv(df_lbl, market)
    
    # Filter codes
    codes = filter_code(df_norm)
    if not codes:
        raise ValueError("No valid codes after filtering")
    
    print(f"Valid codes: {len(codes)}")
    
    # Build industry matrix
    ind_mat = build_industry_matrix(market, codes, mode="identity")
    
    # Get dates in the target month only
    target_month = next_start.strftime("%Y-%m")
    all_dates = sorted(df_norm["dt"].unique().tolist())
    target_dates = [d for d in all_dates if d.startswith(target_month)]
    
    if not target_dates:
        raise ValueError(f"No trading days found for {target_month}")
    
    print(f"Building {len(target_dates)} pkl files for {target_month}...")
    
    # Save daily pkl files
    saved_count = 0
    for dt in target_dates:
        success = save_daily_pkl(
            dt=dt,
            df_all=df_norm,
            codes=codes,
            market=market,
            horizon=horizon,
            relation_type=relation_type,
            lookback=lookback,
            threshold=threshold,
            norm=norm,
            industry_mat=ind_mat,
        )
        if success:
            saved_count += 1
            print(f"  Saved {dt}.pkl")
    
    print(f"Done. Saved {saved_count} pkl files for {target_month}")
    return target_month


def run(args):
    """Entry point for CLI and programmatic calls."""
    return fetch_latest_month_data(
        market=args.market,
        horizon=int(args.horizon) if hasattr(args, 'horizon') else 1,
        relation_type=getattr(args, 'relation_type', 'hy'),
        tickers_file=getattr(args, 'tickers_file', 'tickers.csv'),
        lookback=getattr(args, 'lookback', 20),
        threshold=getattr(args, 'threshold', 0.5),
        norm=not getattr(args, 'disable_norm', False),
        stream=None
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch latest month's data from yfinance")
    parser.add_argument("--market", default="custom", help="Market name")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--relation_type", default="hy", help="Relation type")
    parser.add_argument("--tickers_file", default="tickers.csv", help="Path to tickers CSV")
    parser.add_argument("--lookback", type=int, default=20, help="Lookback window")
    parser.add_argument("--threshold", type=float, default=0.5, help="Correlation threshold")
    parser.add_argument("--disable_norm", action="store_true", help="Disable normalization")
    
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
