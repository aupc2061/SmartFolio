"""
Kafka Producers for Streaming Pipeline

- StockProducer: Sends OHLCV stock data to stock_data topic
- UserProducer: Sends user data with encoded images to user_data topic
"""

from .stock_producer import StockProducer
from .user_producer import UserProducer

__all__ = ["StockProducer", "UserProducer"]
