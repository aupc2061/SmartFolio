"""
Kafka Consumers using Pathway integration.

- StockConsumer: Pathway-based consumer for stock data
- UserConsumer: Pathway-based consumer with risk scoring
- FineTuneLoop: Background loop that triggers streaming fine-tunes
"""

from .stock_consumer import StockConsumer
from .user_consumer import UserConsumer
# from .fine_tune_loop import FineTuneLoop

__all__ = ["StockConsumer", "UserConsumer", "FineTuneLoop"]
