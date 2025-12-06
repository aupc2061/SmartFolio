#!/usr/bin/env python3
"""
Kafka consumer + Pathway streaming pipeline.

Consumes OHLCV JSON messages (schema: {"date": iso, "data": json.dumps({ticker: {Open, High, Low, Close, Adj Close, Volume}})})
from the stock topic, computes display features (daily_change, 1m trend, volatility,
sector_volatility, risk_label), and writes the latest enriched snapshot to disk
for fast API access.

Output: a rolling Parquet file updated on each window flush (default: ./display_data/stream_snapshot.parquet)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pathway as pw
import yfinance as yf

# Type alias for exploded message rows
# (datetime_str, ticker, sector, industry, open, high, low, close, adj_close, volume)
RowType = List[Tuple[str, str, str, str, float, float, float, float, float, float]]

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming.config import (
    KAFKA_BROKER_STOCK_1,
    STOCK_DATA_TOPIC_1,
    TICKERS_PATH,
)


class _MessageSchema(pw.Schema):
    date: str
    data: str


def load_sector_map(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        sector_map = {}
        for _, row in df.iterrows():
            t = row.get("ticker") or row.get("kdcode")
            if not t:
                continue
            sector_map[str(t)] = {
                "sector": row.get("sector", "Unknown"),
                "industry": row.get("industry", "Unknown"),
            }
        return sector_map
    return {}


def enrich_from_yf(ticker: str) -> Dict[str, str]:
    try:
        info = yf.Ticker(ticker).info
        return {
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
        }
    except Exception:
        return {"sector": "Unknown", "industry": "Unknown"}


def parse_args():
    p = argparse.ArgumentParser(description="Consume stock Kafka stream and materialize display snapshot.")
    p.add_argument("--broker", default=KAFKA_BROKER_STOCK_1, help="Kafka bootstrap servers")
    p.add_argument("--topic", default=STOCK_DATA_TOPIC_1, help="Kafka topic to consume")
    p.add_argument("--sector-map", default=None, help="Optional CSV with columns ticker/kdcode,sector[,industry]")
    p.add_argument("--lookback-days", type=int, default=21, help="Window for trend/volatility")
    p.add_argument("--output-path", default="display_data/stream_snapshot.csv", help="Path to write latest snapshot")
    p.add_argument("--flush-interval", type=int, default=60, help="Seconds between snapshot flushes")
    return p.parse_args()


def build_pipeline(args):
    # Kafka source (json messages)
    src = pw.io.kafka.read(
        rdkafka_settings={
            "bootstrap.servers": args.broker,
            "group.id": "display-consumer",
            "auto.offset.reset": "earliest",
        },
        topic=args.topic,
        format="json",
        schema=_MessageSchema,
        autocommit_duration_ms=1_000,
    )

    sector_map = load_sector_map(args.sector_map)

    def explode_message(date: str, data_json: str):
        """Explode a single Kafka message into rows per ticker."""
        payload = json.loads(data_json)
        rows = []
        for ticker, ohlcv in payload.items():
            meta = sector_map.get(ticker) or enrich_from_yf(ticker)
            rows.append(
                (
                    date,  # Keep as string, convert later
                    ticker,
                    meta.get("sector", "Unknown"),
                    meta.get("industry", "Unknown"),
                    float(ohlcv.get("Open", 0.0)),
                    float(ohlcv.get("High", 0.0)),
                    float(ohlcv.get("Low", 0.0)),
                    float(ohlcv.get("Close", 0.0)),
                    float(ohlcv.get("Adj Close", ohlcv.get("Close", 0.0))),
                    float(ohlcv.get("Volume", 0.0)),
                )
            )
        return rows

    # Use apply_with_type to specify the return type explicitly
    exploded = (
        src.select(rows=pw.apply_with_type(explode_message, RowType, pw.this.date, pw.this.data))
        .flatten(pw.this.rows)
        .select(
            dt_str=pw.this.rows[0],
            kdcode=pw.this.rows[1],
            sector=pw.this.rows[2],
            industry=pw.this.rows[3],
            open=pw.this.rows[4],
            high=pw.this.rows[5],
            low=pw.this.rows[6],
            close=pw.this.rows[7],
            prev_close=pw.this.rows[8],
            volume=pw.this.rows[9],
        )
        .with_columns(
            dt=pw.apply_with_type(
                lambda s: datetime.datetime.fromisoformat(s),
                pw.DateTimeNaive,
                pw.this.dt_str
            ),
        )
        .without(pw.this.dt_str)
        .with_columns(
            daily_change=(pw.this.close / (pw.this.prev_close + 1e-8)) - 1.0,
        )
    )

    # Rolling window per ticker
    window = pw.temporal.intervals_over(
        at=exploded.dt,
        lower_bound=-datetime.timedelta(days=args.lookback_days - 1),
        upper_bound=datetime.timedelta(0),
    )
    rolled = (
        exploded.windowby(exploded.dt, window=window, instance=exploded.kdcode)
        .reduce(
            kdcode=pw.this._pw_instance,
            dt=pw.this._pw_window_end,
            sector=pw.reducers.latest(pw.this.sector),
            industry=pw.reducers.latest(pw.this.industry),
            close_first=pw.reducers.earliest(pw.this.close),
            close_last=pw.reducers.latest(pw.this.close),
            dc_sum=pw.reducers.sum(pw.coalesce(pw.this.daily_change, 0.0)),
            dc_sumsq=pw.reducers.sum(
                pw.coalesce(pw.this.daily_change, 0.0) * pw.coalesce(pw.this.daily_change, 0.0)
            ),
            n=pw.reducers.count(),
        )
        .select(
            kdcode=pw.this.kdcode,
            dt=pw.this.dt,
            sector=pw.this.sector,
            industry=pw.this.industry,
            trend_1m=(pw.coalesce(pw.this.close_last, 0.0) - pw.coalesce(pw.this.close_first, 0.0)) / (pw.coalesce(pw.this.close_first, 0.0) + 1e-8),
            volatility=pw.apply_with_type(
                lambda s, ss, n: ((ss / max(n, 1)) - (s / max(n, 1)) ** 2) ** 0.5 * (252**0.5),
                float,
                pw.this.dc_sum,
                pw.this.dc_sumsq,
                pw.this.n,
            ),
        )
    )

    # Latest OHLCV per ticker
    latest = exploded.groupby(pw.this.kdcode).reduce(
        kdcode=pw.this.kdcode,
        dt=pw.reducers.latest(pw.this.dt),
        sector=pw.reducers.latest(pw.this.sector),
        industry=pw.reducers.latest(pw.this.industry),
        close=pw.reducers.latest(pw.this.close),
        open=pw.reducers.latest(pw.this.open),
        high=pw.reducers.latest(pw.this.high),
        low=pw.reducers.latest(pw.this.low),
        prev_close=pw.reducers.latest(pw.this.prev_close),
        volume=pw.reducers.latest(pw.this.volume),
        daily_change=pw.reducers.latest(pw.this.daily_change),
    )

    joined = latest.join_left(rolled, pw.left.kdcode == pw.right.kdcode).select(
        dt=pw.coalesce(pw.right.dt, pw.left.dt),
        kdcode=pw.coalesce(pw.right.kdcode, pw.left.kdcode),
        sector=pw.coalesce(pw.right.sector, pw.left.sector),
        industry=pw.coalesce(pw.right.industry, pw.left.industry),
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

    sector_vol = (
        joined.groupby(joined.dt, joined.sector)
        .reduce(
            dt=pw.this.dt,
            sector=pw.this.sector,
            sector_volatility=pw.reducers.avg(pw.coalesce(joined.volatility, 0.0)),
        )
        .with_id_from(pw.this.dt, pw.this.sector)
    )

    joined_with_key = joined.with_id_from(joined.dt, joined.sector)

    final = joined_with_key.join_left(
        sector_vol,
        pw.left.id == pw.right.id,
    ).select(
        dt=pw.left.dt,
        kdcode=pw.left.kdcode,
        sector=pw.left.sector,
        industry=pw.left.industry,
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
    ).select(
        **pw.this,
        risk_label=pw.apply_with_type(
            lambda r, s: "High" if r is not None and s is not None and r >= 1.2 * s
            else ("Low" if r is not None and s is not None and r <= 0.8 * s else "Medium"),
            str,
            pw.this.volatility,
            pw.this.sector_volatility,
        ),
    )

    # Streaming CSV sink for API to read latest snapshot
    out_path = Path(args.output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pw.io.csv.write(final, str(out_path))

    return final


def main():
    args = parse_args()
    build_pipeline(args)
    pw.run(monitoring_level=pw.MonitoringLevel.NONE)


if __name__ == "__main__":
    main()
