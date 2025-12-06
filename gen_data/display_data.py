import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Optional dependency for fetching sectors
try:
    import yfinance as yf
except Exception:
    yf = None

# Paths
DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset_default"))
DISPLAY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "display_data"))
TREND_LOOKBACK_DAYS = 21

def get_sector_map(tickers, market="custom"):
    """
    Fetch sector and industry info (cached in dataset_default/sector_map.csv).
    Returns (sector_map, industry_map).
    """
    cache_path = os.path.join(DATASET_ROOT, "sector_map.csv")
    
    # 1. Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading cached sector/industry map from {cache_path}...")
        try:
            cached = pd.read_csv(cache_path)
            sector_map = cached.set_index("ticker")["sector"].to_dict()
            industry_map = (
                cached.set_index("ticker")["industry"].to_dict()
                if "industry" in cached.columns
                else {t: "Unknown" for t in tickers}
            )
            return sector_map, industry_map
        except Exception:
            print("Cache corrupt, refetching...")

    # 2. Fetch from Yahoo Finance
    if yf is None:
        print("Warning: yfinance not installed. Sectors/industries will be 'Unknown'.")
        return ({t: "Unknown" for t in tickers}, {t: "Unknown" for t in tickers})

    print(f"Fetching fresh sector/industry data for {len(tickers)} tickers...")
    sector_map = {}
    industry_map = {}
    for t in tqdm(tickers):
        try:
            info = yf.Ticker(t).info
            industry_map[t] = info.get("industry", "Unknown")
            sector_map[t] = info.get("sector", "Unknown")
        except Exception:
            sector_map[t] = "Unknown"
            industry_map[t] = "Unknown"
    
    # 3. Save to cache
    os.makedirs(DATASET_ROOT, exist_ok=True)
    pd.DataFrame(
        {
            "ticker": list(sector_map.keys()),
            "sector": list(sector_map.values()),
            "industry": [industry_map.get(t, "Unknown") for t in sector_map.keys()],
        }
    ).to_csv(cache_path, index=False)
    print(f"Saved sector/industry map to {cache_path}")
    return sector_map, industry_map

def _trend_over_window(window: np.ndarray) -> float:
    if len(window) == 0: return np.nan
    return (window[-1] - window[0]) / (window[0] + 1e-8)

def add_engineered_features(df: pd.DataFrame, trend_lookback: int = TREND_LOOKBACK_DAYS) -> pd.DataFrame:
    """Restore the engineered features required by frontend."""
    df = df.copy()
    df = df.sort_values(["kdcode", "dt"])
    
    # Daily Change
    df["daily_change"] = df["close"] / df["prev_close"] - 1
    
    # 1-Month Trend
    df["trend_1m"] = df.groupby("kdcode")["close"].transform(
        lambda x: x.rolling(window=trend_lookback, min_periods=1).apply(_trend_over_window, raw=True)
    )
    
    return df

def classify_risk(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Add Volatility and Risk Labels."""
    print("Calculating Volatility & Risk Profiles...")
    
    # 1. Calculate Volatility (Annualized Rolling Std Dev)
    # We use the 'daily_change' we just calculated
    df["volatility"] = df.groupby("kdcode")["daily_change"].transform(
        lambda x: x.rolling(lookback).std()
    ) * np.sqrt(252)
    
    # 2. Calculate Sector Benchmarks
    # Group by [Date, Sector] to get avg volatility for that sector on that day
    sector_stats = df.groupby(["dt", "sector"])["volatility"].mean().reset_index()
    sector_stats = sector_stats.rename(columns={"volatility": "sector_volatility"})
    
    # Merge back
    df = pd.merge(df, sector_stats, on=["dt", "sector"], how="left")
    
    # 3. Risk Label Logic
    # Ratio = Stock / Sector
    df["risk_ratio"] = df["volatility"] / (df["sector_volatility"] + 1e-6)
    
    conditions = [
        (df["risk_ratio"] >= 1.2), # High Risk
        (df["risk_ratio"] <= 0.8)  # Low Risk
    ]
    choices = ["High", "Low"]
    df["risk_label"] = np.select(conditions, choices, default="Medium")
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="custom")
    parser.add_argument("--lookback", type=int, default=20, help="Window for volatility calc")
    args = parser.parse_args()

    # 1. Load Existing Data (The 'org' CSV from build_dataset_yf)
    org_file = os.path.join(DATASET_ROOT, f"{args.market}_org.csv")
    if not os.path.exists(org_file):
        raise FileNotFoundError(f"Could not find {org_file}. Please run build_dataset_yf.py first.")
    
    print(f"Loading dataset from {org_file}...")
    df = pd.read_csv(org_file)
    df["dt"] = pd.to_datetime(df["dt"])

    # 2. Add Sectors (Fetch once, reuse forever)
    tickers = df["kdcode"].unique().tolist()
    sector_map, industry_map = get_sector_map(tickers, args.market)
    df["sector"] = df["kdcode"].map(sector_map).fillna("Unknown")
    df["industry"] = df["kdcode"].map(industry_map).fillna("Unknown")

    # 3. Re-Engineer Features (Daily Change, Trends)
    # We re-run this to guarantee they exist even if org_csv was raw
    df = add_engineered_features(df)

    # 4. Add Risk Ratings
    df = classify_risk(df, args.lookback)

    # 5. Save Display Data
    os.makedirs(DISPLAY_ROOT, exist_ok=True)
    out_path = os.path.join(DISPLAY_ROOT, f"{args.market}_display.csv")
    
    # Final Columns for Frontend
    # Preserving all original features + adding new ones
    target_cols = [
        "dt", "kdcode", "sector", "industry",
        "close", "open", "high", "low", "prev_close", "volume", 
        "daily_change", "trend_1m", 
        "volatility", "sector_volatility", "risk_label"
    ]
    
    # Filter to ensure we only select columns that exist
    final_cols = [c for c in target_cols if c in df.columns]
    
    # Formatting
    df_final = df[final_cols].copy()
    df_final["dt"] = df_final["dt"].dt.strftime("%Y-%m-%d")
    
    df_final.dropna().to_csv(out_path, index=False)
    print(f"Generated Display Data: {out_path}")
    print(f"Features included: {len(final_cols)}")

if __name__ == "__main__":
    main()
