"""
Main entry point for running Kafka Consumers with Pathway.

Runs:
- Stock consumer (Pathway-based)
- User consumer (Pathway-based with risk scoring)
- Finetune manager (12 worker threads)
"""

import logging
import signal
import threading
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streaming.consumer.stock_consumer import StockConsumer, create_stock_consumer_pipeline
from streaming.consumer.user_consumer import UserConsumer, create_user_consumer_pipeline
# from streaming.consumer.fine_tune_loop import FineTuneLoop
from streaming.shared.state import SharedState

import pathway as pw

logger = logging.getLogger(__name__)


class ConsumerManager:
    """
    Manages Pathway-based Kafka consumers and finetune workers.
    
    Runs:
    - Stock consumer pipeline
    - User consumer pipeline  
    - 12 finetune worker threads
    """
    
    def __init__(self):
        self.stock_consumer = StockConsumer()
        self.user_consumer = UserConsumer()
        # self.finetune_loop = FineTuneLoop()
        
        self._pathway_thread: Optional[threading.Thread] = None
        self._running = False
        self._state = SharedState()
    
    def start(self):
        """Start all consumers and finetune workers."""
        self._running = True
        
        # Start finetune workers first
        logger.info("Starting fine-tune loop...")
        # self.finetune_loop.start()
        
        # Start Pathway consumers in a thread
        logger.info("Starting Pathway consumers...")
        self._pathway_thread = threading.Thread(
            target=self._run_pathway,
            name="PathwayConsumerThread",
            daemon=True,
        )
        self._pathway_thread.start()
        
        logger.info("All consumers started")
    
    def _run_pathway(self):
        """Run Pathway pipelines in a single runtime."""
        try:
            # Build combined pipeline
            stock_table = self.stock_consumer.build_pipeline()
            user_table = self.user_consumer.build_pipeline()
            
            # Null outputs (side effects handled in UDFs)
            pw.io.null.write(stock_table)
            pw.io.null.write(user_table)
            
            # Run Pathway (blocks until stopped)
            logger.info("Pathway runtime starting...")
            pw.run(monitoring_level=pw.MonitoringLevel.NONE)
            
        except Exception as e:
            logger.exception(f"Pathway runtime error: {e}")
    
    def stop(self):
        """Stop all consumers gracefully."""
        logger.info("Stopping consumers...")
        self._running = False
        
        # Stop finetune workers
        # self.finetune_loop.stop()
        
        # Note: Pathway doesn't have a clean stop mechanism
        # The thread will be killed when the process exits
        
        logger.info("Consumers stopped")
    
    def wait(self):
        """Wait for pathway thread to complete."""
        if self._pathway_thread:
            self._pathway_thread.join()
    
    def get_status(self) -> dict:
        """Get status of all components."""
        return {
            "pathway_running": (
                self._pathway_thread is not None and 
                self._pathway_thread.is_alive()
            ),
            # "finetune_loop_alive": self.finetune_loop.is_alive(),
            "stock_buffer": {
                "current_month": self._state.stock_buffer.current_month,
                "buffer_size": len(self._state.stock_buffer._buffer),
            },
            "user_risk_data_count": len(self._state.get_all_user_risk_data()),
        }


def run_all_consumers():
    """Run stock consumer, user consumer, and finetune manager together."""
    manager = ConsumerManager()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        manager.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        manager.start()
        
        logger.info("Consumers running. Press Ctrl+C to stop.")
        
        # Monitor loop
        while manager._running:
            time.sleep(30)
            
            # Print status
            status = manager.get_status()
            logger.info(f"Consumer status: {status}")
            
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()


def run_stock_consumer_only():
    """Run only the stock consumer."""
    from streaming.consumer.stock_consumer import run_stock_consumer
    run_stock_consumer()


def run_user_consumer_only():
    """Run only the user consumer."""
    from streaming.consumer.user_consumer import run_user_consumer
    run_user_consumer()


# def run_finetune_manager_only():
#     """Run only the finetune manager (single pass)."""
#     from streaming.consumer.fine_tune_loop import FineTuneLoop
    
#     loop = FineTuneLoop()
#     loop._process_once()


def main():
    """Main entry point with CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Kafka Consumers with Pathway Integration"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "stock", "user", "finetune"],
        default="all",
        help="Which consumers to run (default: all)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run selected mode
    if args.mode == "all":
        run_all_consumers()
    elif args.mode == "stock":
        run_stock_consumer_only()
    elif args.mode == "user":
        run_user_consumer_only()
    # elif args.mode == "finetune":
    #     run_finetune_manager_only()


if __name__ == "__main__":
    main()
