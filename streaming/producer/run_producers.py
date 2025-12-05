"""
Main entry point for running Kafka producers.

Runs both stock and user data producers in separate threads.
"""

import logging
import signal
import threading
import time
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streaming.producer.stock_producer import StockProducer
from streaming.producer.user_producer import UserProducer
from streaming.config import PRODUCER_DELAY_STOCK, PRODUCER_DELAY_USER

logger = logging.getLogger(__name__)


class ProducerManager:
    """
    Manages multiple Kafka producers in separate threads.
    """
    
    def __init__(
        self,
        stock_delay: float = PRODUCER_DELAY_STOCK,
        user_delay: float = PRODUCER_DELAY_USER,
        loop: bool = True,
    ):
        self.stock_delay = stock_delay
        self.user_delay = user_delay
        self.loop = loop
        
        self.stock_producer = StockProducer(delay=stock_delay)
        self.user_producer = UserProducer(delay=user_delay)
        
        self.stock_thread: Optional[threading.Thread] = None
        self.user_thread: Optional[threading.Thread] = None
        
        self._running = False
    
    def start(self):
        """Start all producer threads."""
        self._running = True
        
        # Connect producers
        if not self.stock_producer.connect():
            logger.error("Failed to connect stock producer")
            return False
        
        if not self.user_producer.connect():
            logger.error("Failed to connect user producer")
            self.stock_producer.disconnect()
            return False
        
        # Start threads
        self.stock_thread = threading.Thread(
            target=self._run_stock_producer,
            name="StockProducerThread",
            daemon=True,
        )
        
        self.user_thread = threading.Thread(
            target=self._run_user_producer,
            name="UserProducerThread",
            daemon=True,
        )
        
        self.stock_thread.start()
        self.user_thread.start()
        
        logger.info("All producer threads started")
        return True
    
    def _run_stock_producer(self):
        """Thread target for stock producer."""
        try:
            self.stock_producer.stream_data(loop=self.loop)
        except Exception as e:
            logger.exception(f"Stock producer error: {e}")
    
    def _run_user_producer(self):
        """Thread target for user producer."""
        try:
            self.user_producer.stream_data(loop=self.loop)
        except Exception as e:
            logger.exception(f"User producer error: {e}")
    
    def stop(self):
        """Stop all producer threads gracefully."""
        logger.info("Stopping all producers...")
        self._running = False
        
        # Signal producers to stop
        self.stock_producer.stop()
        self.user_producer.stop()
        
        # Wait for threads to finish
        if self.stock_thread and self.stock_thread.is_alive():
            self.stock_thread.join(timeout=5)
        
        if self.user_thread and self.user_thread.is_alive():
            self.user_thread.join(timeout=5)
        
        # Disconnect
        self.stock_producer.disconnect()
        self.user_producer.disconnect()
        
        logger.info("All producers stopped")
    
    def wait(self):
        """Wait for all producer threads to complete."""
        if self.stock_thread:
            self.stock_thread.join()
        if self.user_thread:
            self.user_thread.join()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Kafka Producers for Streaming Pipeline")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop through data continuously"
    )
    parser.add_argument(
        "--stock-delay",
        type=float,
        default=PRODUCER_DELAY_STOCK,
        help=f"Delay between stock messages (default: {PRODUCER_DELAY_STOCK}s)"
    )
    parser.add_argument(
        "--user-delay",
        type=float,
        default=PRODUCER_DELAY_USER,
        help=f"Delay between user messages (default: {PRODUCER_DELAY_USER}s)"
    )
    parser.add_argument(
        "--stock-only",
        action="store_true",
        help="Run only stock producer"
    )
    parser.add_argument(
        "--user-only",
        action="store_true",
        help="Run only user producer"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Handle individual producer runs
    if args.stock_only:
        from streaming.producer.stock_producer import run_stock_producer
        run_stock_producer(loop=args.loop, delay=args.stock_delay)
        return
    
    if args.user_only:
        from streaming.producer.user_producer import run_user_producer
        run_user_producer(loop=args.loop, delay=args.user_delay)
        return
    
    # Run both producers
    manager = ProducerManager(
        stock_delay=args.stock_delay,
        user_delay=args.user_delay,
        loop=args.loop,
    )
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        manager.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start and wait
    if manager.start():
        logger.info("Producers running. Press Ctrl+C to stop.")
        try:
            manager.wait()
        except KeyboardInterrupt:
            pass
        finally:
            manager.stop()
    else:
        logger.error("Failed to start producers")


if __name__ == "__main__":
    main()
