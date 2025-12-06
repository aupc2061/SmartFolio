#!/usr/bin/env python3
"""
Streaming version of display_data.py using Pathway.

Assumes an upstream producer (e.g., yfinance poller) publishes OHLCV ticks to a Kafka
topic as JSON lines with fields:
  - dt (ISO datetime string, e.g., "2025-01-02T09:15:00")
  - ticker (str)
  - open, high, low, close, prev_close, volume (float)
  - sector (optional str; if missing, a fallback sector_map.csv can be provided)

This pipeline:
  1) Consumes the stream.
  2) Computes daily_change and a 1‑month trend (window=lookback_days).
  3) Computes rolling volatility (annualized) per ticker.
  4) Computes sector-average volatility per day and a risk_label.
  5) Emits enriched rows to a Kafka topic or to a filesystem sink.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from typing import Dict, Optional

import pandas as pd
import pathway as pw


def load_sector_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sector map not found: {path}")
    df = pd.read_csv(path)
    if "ticker" not in df.columns or "sector" not in df.columns:
        raise ValueError("Sector map must have columns: ticker, sector")
    return dict(zip(df["ticker"], df["sector"]))


def build_pipeline(args: argparse.Namespace):
    # Kafka source: expects JSON lines
    src = pw.io.kafka.read(
        brokers=args.kafka_brokers,
        topic=args.kafka_topic_in,
        format="json",
        autocommit_duration_ms=1_000,
    )

    sector_map = load_sector_map(args.sector_map)

    def parse_row(row: dict):
        # Minimal parsing/typing
        dt_str = row.get("dt")
        dt_val = datetime.datetime.fromisoformat(dt_str) if dt_str else None
        ticker = row.get("ticker")
        sector = row.get("sector") or sector_map.get(ticker, "Unknown")
        def f(key, default=0.0):
            try:
                return float(row.get(key, default))
            except Exception:
                return default
        return {
            "dt": dt_val,
            "ticker": ticker,
            "sector": sector,
            "open": f("open"),
            "high": f("high"),
            "low": f("low"),
            "close": f("close"),
            "prev_close": f("prev_close"),
            "volume": f("volume"),
        }

    parsed = src.select(**pw.apply(parse_row, pw.this.value))

    # Core features
    enriched = parsed.select(
        dt=pw.this.dt,
        ticker=pw.this.ticker,
        sector=pw.this.sector,
        open=pw.this.open,
        high=pw.this.high,
        low=pw.this.low,
        close=pw.this.close,
        prev_close=pw.this.prev_close,
        volume=pw.this.volume,
        daily_change=(pw.this.close / (pw.this.prev_close + 1e-8)) - 1.0,
    )

    # Rolling window per ticker: trend and volatility
    window = pw.temporal.intervals_over(
        at=enriched.dt,
        lower_bound=-datetime.timedelta(days=args.lookback_days - 1),
        upper_bound=datetime.timedelta(0),
    )
    rolled = (
        enriched.windowby(enriched.dt, window=window, instance=enriched.ticker)
        .reduce(
            ticker=pw.this._pw_instance,
            dt=pw.this._pw_window_end,
            sector=pw.reducers.last(enriched.sector),
            close_first=pw.reducers.first(enriched.close),
            close_last=pw.reducers.last(enriched.close),
            dc_sum=pw.reducers.sum(enriched.daily_change),
            dc_sumsq=pw.reducers.sum(enriched.daily_change * enriched.daily_change),
            n=pw.reducers.count(),
        )
        .select(
            ticker=pw.this.ticker,
            dt=pw.this.dt,
            sector=pw.this.sector,
            trend_1m=(pw.this.close_last - pw.this.close_first)
            / (pw.this.close_first + 1e-8),
            volatility=pw.apply(
                lambda s, ss, n: ((ss / max(n, 1)) - (s / max(n, 1)) ** 2) ** 0.5
                * (252**0.5),
                pw.this.dc_sum,
                pw.this.dc_sumsq,
                pw.this.n,
            ),
        )
    )

    # Join back with latest row to keep prices/volume/daily_change
    latest = enriched.groupby(enriched.ticker).reduce(
        ticker=pw.this.ticker,
        dt=pw.reducers.last(enriched.dt),
        sector=pw.reducers.last(enriched.sector),
        close=pw.reducers.last(enriched.close),
        open=pw.reducers.last(enriched.open),
        high=pw.reducers.last(enriched.high),
        low=pw.reducers.last(enriched.low),
        prev_close=pw.reducers.last(enriched.prev_close),
        volume=pw.reducers.last(enriched.volume),
        daily_change=pw.reducers.last(enriched.daily_change),
    )

    joined = latest.join_left(rolled, pw.left.ticker == pw.right.ticker).select(
        dt=pw.coalesce(pw.right.dt, pw.left.dt),
        ticker=pw.left.ticker,
        sector=pw.coalesce(pw.right.sector, pw.left.sector),
        close=pw.left.close,
        open=pw.left.open,
        high=pw.left.high,
        low=pw.left.low,
        prev_close=pw.left.prev_close,
        volume=pw.left.volume,
        daily_change=pw.left.daily_change,
        trend_1m=pw.right.trend_1m,
        volatility=pw.right.volatility,
    )

    # Sector volatility per day
    sector_vol = (
        joined.groupby(joined.dt, joined.sector)
        .reduce(
            dt=pw.this.dt,
            sector=pw.this.sector,
            sector_volatility=pw.reducers.mean(joined.volatility),
        )
    )

    final = joined.join_left(
        sector_vol,
        (pw.left.dt == pw.right.dt) & (pw.left.sector == pw.right.sector),
    ).select(
        dt=pw.left.dt,
        ticker=pw.left.ticker,
        sector=pw.left.sector,
        close=pw.left.close,
        open=pw.left.open,
        high=pw.left.high,
        low=pw.left.low,
        prev_close=pw.left.prev_close,
        volume=pw.left.volume,
        daily_change=pw.left.daily_change,
        trend_1m=pw.left.trend_1m,
        volatility=pw.left.volatility,
        sector_volatility=pw.right.sector_volatility,
        risk_ratio=pw.left.volatility / (pw.right.sector_volatility + 1e-6),
    ).select(
        **pw.this,
        risk_label=pw.apply(
            lambda r: "High" if r >= 1.2 else ("Low" if r <= 0.8 else "Medium"),
            pw.this.risk_ratio,
        ),
    )

    if args.kafka_topic_out:
        pw.io.kafka.write(
            final,
            brokers=args.kafka_brokers,
            topic=args.kafka_topic_out,
            format="json",
        )
    if args.output_dir:
        pw.io.csv.write(final, args.output_dir, mode="streaming")


def parse_args():
    p = argparse.ArgumentParser(description="Pathway streaming display data builder")
    p.add_argument("--kafka-brokers", required=True, help="Comma-separated Kafka bootstrap servers")
    p.add_argument("--kafka-topic-in", required=True, help="Input topic with OHLCV JSON events")
    p.add_argument("--kafka-topic-out", default=None, help="Optional output topic for enriched rows")
    p.add_argument("--output-dir", default=None, help="Optional streaming CSV sink directory")
    p.add_argument("--lookback-days", type=int, default=21, help="Window for trend/volatility")
    p.add_argument("--sector-map", default=None, help="CSV with columns ticker,sector for fallback mapping")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_pipeline(args)
    pw.run()
