#!/usr/bin/env python3
"""
Live yfinance producer (Kafka).

Polls yfinance every minute (1m interval) for the configured tickers and pushes new
OHLCV bars to Kafka in the same schema used by StockDataSchema:
  { "date": "<ISO timestamp>", "data": "<json mapping ticker -> OHLCV>" }

Notes:
- Uses period="2d" to avoid yfinance 7‑day 1m limit issues; filters out already-sent bars.
- Maintains last_sent per ticker to prevent re-emitting old bars.
- Flushes after each polling cycle to avoid unbounded buffers.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf
from kafka import KafkaProducer
from kafka.errors import KafkaError

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming.config import (
    KAFKA_BROKER_STOCK,
    STOCK_DATA_TOPIC,
    TICKERS_PATH,
)

logger = logging.getLogger(__name__)


def _load_tickers(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Tickers file not found: {path}")
    df = pd.read_csv(path)
    if "ticker" in df.columns:
        tickers = df["ticker"].dropna().astype(str).tolist()
    else:
        tickers = df.iloc[:, 0].dropna().astype(str).tolist()
    return tickers


class YFLiveProducer:
    def __init__(
        self,
        tickers: list[str],
        broker: str = KAFKA_BROKER_STOCK,
        topic: str = STOCK_DATA_TOPIC,
        poll_interval: int = 60,
    ):
        self.tickers = tickers
        self.broker = broker
        self.topic = topic
        self.poll_interval = poll_interval
        self.producer: Optional[KafkaProducer] = None
        self.last_sent: Dict[str, pd.Timestamp] = {}

    def connect(self) -> bool:
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.broker,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                retry_backoff_ms=500,
            )
            logger.info(f"Connected to Kafka broker {self.broker}")
            return True
        except KafkaError as e:
            logger.error(f"Kafka connection failed: {e}")
            return False

    def disconnect(self):
        if self.producer:
            self.producer.flush()
            self.producer.close()
            self.producer = None
            logger.info("Kafka producer closed.")

    def _fetch_latest(self) -> pd.DataFrame:
        df = yf.download(
            tickers=self.tickers,
            period="2d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index).tz_localize(None).tz_localize(timezone.utc)
        return df

    def _format_message(self, ts: pd.Timestamp, row) -> Optional[dict]:
        data = {}
        for ticker in self.tickers:
            if ticker not in row:
                continue
            series = row[ticker]
            try:
                ohlcv = {
                    "Open": float(series.get("Open", 0.0)),
                    "High": float(series.get("High", 0.0)),
                    "Low": float(series.get("Low", 0.0)),
                    "Close": float(series.get("Close", 0.0)),
                    "Adj Close": float(series.get("Adj Close", series.get("Close", 0.0))),
                    "Volume": float(series.get("Volume", 0.0)),
                }
                if any(v != 0 for v in ohlcv.values()):
                    data[ticker] = ohlcv
            except Exception:
                continue
        if not data:
            return None
        return {
            "date": ts.isoformat(),
            "data": json.dumps(data),
        }

    def run(self):
        if not self.producer and not self.connect():
            return

        logger.info(
            f"Starting yfinance streaming for {len(self.tickers)} tickers, interval=1m, poll={self.poll_interval}s"
        )
        try:
            while True:
                df = self._fetch_latest()
                if df.empty:
                    logger.warning("No data returned from yfinance poll.")
                    time.sleep(self.poll_interval)
                    continue

                for ts, row in df.iterrows():
                    already_sent = any(
                        ts <= self.last_sent.get(t, pd.Timestamp.min.tz_localize(timezone.utc))
                        for t in self.tickers
                    )
                    if already_sent:
                        continue
                    msg = self._format_message(ts, row)
                    if not msg:
                        continue
                    key = msg["date"]
                    try:
                        fut = self.producer.send(self.topic, key=key, value=msg)
                        fut.get(timeout=10)
                        for t in json.loads(msg["data"]).keys():
                            self.last_sent[t] = ts
                    except KafkaError as e:
                        logger.error(f"Failed to send message for {key}: {e}")

                self.producer.flush()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.disconnect()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stream live yfinance 1m data to Kafka.")
    parser.add_argument("--tickers-file", default=str(TICKERS_PATH), help="CSV with tickers column")
    parser.add_argument("--broker", default=KAFKA_BROKER_STOCK, help="Kafka bootstrap servers")
    parser.add_argument("--topic", default=STOCK_DATA_TOPIC, help="Kafka topic to publish to")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between yfinance polls")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    tickers = _load_tickers(Path(args.tickers_file))
    producer = YFLiveProducer(
        tickers=tickers,
        broker=args.broker,
        topic=args.topic,
        poll_interval=args.poll_interval,
    )
    producer.run()


if __name__ == "__main__":
    main()
