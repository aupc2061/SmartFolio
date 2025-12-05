"""
Stock Data Producer

Reads OHLCV data from ohlcv_raw.csv and streams to Kafka topic.
Simulates real-time data by adding delay between messages.
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streaming.config import (
    KAFKA_BROKER_STOCK,
    STOCK_DATA_TOPIC,
    OHLCV_RAW_PATH,
    PRODUCER_DELAY_STOCK,
)
from streaming.shared.utils import load_ohlcv_dataframe

logger = logging.getLogger(__name__)


class StockProducer:
    """
    Kafka producer for stock OHLCV data.
    
    Reads from ohlcv_raw.csv (MultiIndex format) and streams each row
    as a JSON message to the stock_data Kafka topic.
    """
    
    def __init__(
        self,
        broker: str = KAFKA_BROKER_STOCK,
        topic: str = STOCK_DATA_TOPIC,
        delay: float = PRODUCER_DELAY_STOCK,
    ):
        self.broker = broker
        self.topic = topic
        self.delay = delay
        self.producer: Optional[KafkaProducer] = None
        self._running = False
        
    def connect(self) -> bool:
        """Initialize Kafka producer connection."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.broker,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                retry_backoff_ms=500,
            )
            logger.info(f"Connected to Kafka broker at {self.broker}")
            return True
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    def disconnect(self):
        """Close Kafka producer connection."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            self.producer = None
            logger.info("Disconnected from Kafka broker")
    
    def send_message(self, key: str, value: dict) -> bool:
        """Send a single message to Kafka topic."""
        if not self.producer:
            logger.error("Producer not connected")
            return False
        
        try:
            future = self.producer.send(self.topic, key=key, value=value)
            # Block for synchronous send (optional, can be async)
            future.get(timeout=10)
            logger.info(f"Sent stock data for key={key}")
            return True
        except KafkaError as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def stream_data(self, csv_path: Optional[Path] = None, loop: bool = False):
        """
        Stream OHLCV data from CSV to Kafka.
        
        Args:
            csv_path: Path to ohlcv_raw.csv (uses default if None)
            loop: If True, restart from beginning after reaching end
        """
        csv_path = csv_path or OHLCV_RAW_PATH
        
        if not csv_path.exists():
            logger.error(f"OHLCV file not found: {csv_path}")
            return
        
        if not self.producer:
            if not self.connect():
                return
        
        self._running = True
        logger.info(f"Starting stock data stream from {csv_path}")
        
        try:
            while self._running:
                # Load DataFrame with MultiIndex handling
                result = load_ohlcv_dataframe(str(csv_path))
                df, tickers = result
                
                if df is None or df.empty:
                    logger.error("Failed to load OHLCV data")
                    break
                
                logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns, {len(tickers)} tickers")
                
                # Stream each row
                for idx, row in df.iterrows():
                    if not self._running:
                        break
                    
                    # Format row as message
                    message = self._format_message(str(idx), row, tickers)
                    if message:
                        # Use date as key for partitioning
                        key = message.get("date", str(idx))
                        
                        if self.send_message(key, message):
                            logger.debug(f"Sent stock data for {key}")
                        else:
                            logger.warning(f"Failed to send data for {key}")
                        
                        # Simulate real-time delay
                        time.sleep(self.delay)
                
                if not loop:
                    break
                    
                logger.info("Restarting stream from beginning...")
                
        except KeyboardInterrupt:
            logger.info("Stock producer interrupted by user")
        except Exception as e:
            logger.exception(f"Error in stock stream: {e}")
        finally:
            self._running = False
            logger.info("Stock data stream ended")
    
    def _format_message(self, date: str, row, tickers: list) -> Optional[dict]:
        """Format a row into a message for Kafka."""
        import math
        try:
            data = {}
            for ticker in tickers:
                try:
                    # Helper to safely convert to float, treating NaN as 0
                    def safe_float(val, default=0.0):
                        if val is None:
                            return default
                        try:
                            f = float(val)
                            return default if math.isnan(f) else f
                        except (ValueError, TypeError):
                            return default
                    
                    ohlcv = {
                        "Open": safe_float(row.get((ticker, "Open"), 0)),
                        "High": safe_float(row.get((ticker, "High"), 0)),
                        "Low": safe_float(row.get((ticker, "Low"), 0)),
                        "Close": safe_float(row.get((ticker, "Close"), 0)),
                        "Adj Close": safe_float(row.get((ticker, "Adj Close"), 0)),
                        "Volume": safe_float(row.get((ticker, "Volume"), 0)),
                    }
                    # Skip if all values are 0 (no valid data for this ticker on this date)
                    if any(v != 0 for v in ohlcv.values()):
                        data[ticker] = ohlcv
                except (ValueError, TypeError):
                    continue
            
            if data:
                return {
                    "date": date,
                    "data": json.dumps(data),
                }
            return None
        except Exception as e:
            logger.error(f"Error formatting message: {e}")
            return None
    
    def stop(self):
        """Stop the streaming loop."""
        self._running = False


def run_stock_producer(loop: bool = False, delay: Optional[float] = None):
    """
    Main entry point for stock producer.
    
    Args:
        loop: Whether to loop through data continuously
        delay: Override default delay between messages
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    producer = StockProducer()
    if delay is not None:
        producer.delay = delay
    
    try:
        producer.stream_data(loop=loop)
    finally:
        producer.disconnect()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock Data Kafka Producer")
    parser.add_argument("--loop", action="store_true", help="Loop through data continuously")
    parser.add_argument(
        "--delay",
        type=float,
        default=PRODUCER_DELAY_STOCK,
        help=f"Delay between stock messages in seconds (default: {PRODUCER_DELAY_STOCK})",
    )
    
    args = parser.parse_args()
    run_stock_producer(loop=args.loop, delay=args.delay)
