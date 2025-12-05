"""
Main entry point for the Streaming Pipeline.

This script runs:
1. Kafka Producers (stock and user data)
2. Pathway-based Kafka Consumers
3. 12 Fine-tuning threads (one per risk group)

Usage:
    # Run everything (producers + consumers)
    python -m streaming.run_all

    # Run only producers
    python -m streaming.run_all --producers-only
    
    # Run only consumers
    python -m streaming.run_all --consumers-only
"""

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming.producer.run_producers import ProducerManager
from streaming.consumer.run_consumers import ConsumerManager
from streaming.config import (
    PRODUCER_DELAY_STOCK,
    PRODUCER_DELAY_USER,
    KAFKA_BROKER,
    STOCK_DATA_TOPIC,
    USER_DATA_TOPIC,
)

logger = logging.getLogger(__name__)


class StreamingPipeline:
    """
    Orchestrates the entire streaming pipeline.
    """
    
    def __init__(
        self,
        run_producers: bool = True,
        run_consumers: bool = True,
        stock_delay: float = PRODUCER_DELAY_STOCK,
        user_delay: float = PRODUCER_DELAY_USER,
        loop_producers: bool = True,
    ):
        self.run_producers = run_producers
        self.run_consumers = run_consumers
        self.stock_delay = stock_delay
        self.user_delay = user_delay
        self.loop_producers = loop_producers
        
        self.producer_manager = None
        self.consumer_manager = None
        self._running = False
    
    def start(self):
        """Start the streaming pipeline."""
        self._running = True
        
        logger.info("=" * 60)
        logger.info("Starting Streaming Pipeline")
        logger.info(f"  Kafka Broker: {KAFKA_BROKER}")
        logger.info(f"  Stock Topic:  {STOCK_DATA_TOPIC}")
        logger.info(f"  User Topic:   {USER_DATA_TOPIC}")
        logger.info("=" * 60)
        
        # Start consumers first (they need to be ready for data)
        if self.run_consumers:
            logger.info("Starting Kafka consumers with Pathway...")
            self.consumer_manager = ConsumerManager()
            self.consumer_manager.start()
            
            # Give consumers time to initialize
            time.sleep(2)
        
        # Start producers
        if self.run_producers:
            logger.info("Starting Kafka producers...")
            self.producer_manager = ProducerManager(
                stock_delay=self.stock_delay,
                user_delay=self.user_delay,
                loop=self.loop_producers,
            )
            self.producer_manager.start()
        
        logger.info("Streaming pipeline is running!")
    
    def stop(self):
        """Stop the streaming pipeline."""
        logger.info("Stopping streaming pipeline...")
        self._running = False
        
        if self.producer_manager:
            self.producer_manager.stop()
        
        if self.consumer_manager:
            self.consumer_manager.stop()
        
        logger.info("Streaming pipeline stopped.")
    
    def wait(self):
        """Wait for pipeline to complete."""
        try:
            while self._running:
                time.sleep(5)
                self._print_status()
        except KeyboardInterrupt:
            pass
    
    def _print_status(self):
        """Print pipeline status."""
        status = []
        
        if self.consumer_manager:
            cm_status = self.consumer_manager.get_status()
            status.append(f"Pathway: {'running' if cm_status['pathway_running'] else 'stopped'}")
            status.append(f"Users processed: {cm_status['user_risk_data_count']}")
            
            active_workers = sum(
                1 for s in cm_status['finetune_workers'].values() 
                if s['is_alive']
            )
            total_finetunes = sum(
                s['finetune_count'] 
                for s in cm_status['finetune_workers'].values()
            )
            status.append(f"Finetune workers: {active_workers}/12 active, {total_finetunes} runs")
        
        logger.info(" | ".join(status))


def main():
    parser = argparse.ArgumentParser(
        description="Run the Streaming Pipeline for Portfolio Management"
    )
    parser.add_argument(
        "--producers-only",
        action="store_true",
        help="Run only the Kafka producers"
    )
    parser.add_argument(
        "--consumers-only",
        action="store_true",
        help="Run only the Kafka consumers"
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
        "--no-loop",
        action="store_true",
        help="Don't loop producers (run data once)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Determine what to run
    run_producers = not args.consumers_only
    run_consumers = not args.producers_only
    
    if args.producers_only and args.consumers_only:
        logger.error("Cannot specify both --producers-only and --consumers-only")
        sys.exit(1)
    
    # Create and run pipeline
    pipeline = StreamingPipeline(
        run_producers=run_producers,
        run_consumers=run_consumers,
        stock_delay=args.stock_delay,
        user_delay=args.user_delay,
        loop_producers=not args.no_loop,
    )
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        pipeline.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        pipeline.start()
        pipeline.wait()
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
