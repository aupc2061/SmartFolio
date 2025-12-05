"""
Stock Data Consumer with Pathway-Kafka Integration.

Uses Pathway's Kafka connector to consume stock OHLCV data,
detect month changes, buffer data, and save to CSV for fine-tuning.
"""

import json
import logging
import os
import threading
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Optional

import pathway as pw
from pathway.io import kafka as pw_kafka

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streaming.config import (
    KAFKA_BROKER_STOCK,
    STOCK_DATA_TOPIC,
    KAFKA_GROUP_STOCK,
    MONTHLY_DATA_DIR,
    SMARTFOLIO_DIR,
)
from streaming.shared.state import SharedState
from streaming.consumer.schemas import StockDataSchema

logger = logging.getLogger(__name__)

# Track active fine-tuning threads
_finetune_lock = threading.Lock()
_finetune_thread: Optional[threading.Thread] = None


def _build_default_args(risk_score: float = 0.5) -> Namespace:
    """
    Build default arguments for fine-tuning, matching main.py defaults.
    
    Args:
        risk_score: Risk score (0.0-1.0) to configure checkpoint paths
        
    Returns:
        Namespace with all required arguments
    """
    # Import here to avoid circular imports
    from streaming.config import get_risk_score_dir, get_baseline_checkpoint
    
    args = Namespace()
    
    # Basic settings
    args.market = "custom"
    args.horizon = "1"
    args.relation_type = "hy"
    args.policy = "HGAT"
    args.model_name = "SmartFolio"
    args.seed = 123
    
    # Device
    import torch
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Graph relation flags
    args.ind_yn = True
    args.pos_yn = True
    args.neg_yn = True
    args.multi_reward = True
    
    # Training parameters
    args.irl_epochs = 50
    args.rl_timesteps = 10000
    args.fine_tune_steps = 5000
    args.batch_size = 512
    args.n_steps = 2048
    args.max_epochs = 10
    args.num_expert_trajectories = 700
    args.lookback = 20
    
    # Risk settings - use the provided risk_score
    args.risk_score = risk_score
    args.dd_base_weight = 1.0
    args.dd_risk_factor = 1.0
    
    # PTR settings
    args.ptr_mode = True
    args.ptr_coef = 0.3
    args.ptr_memory_size = 1000
    args.ptr_priority_type = "max"
    
    # Paths - use dynamic risk-score-based directories (matching main.py convention)
    args.save_dir = get_risk_score_dir(str(SMARTFOLIO_DIR / "checkpoints"), risk_score)
    args.baseline_checkpoint = str(get_baseline_checkpoint(risk_score))
    args.tickers_file = str(SMARTFOLIO_DIR / "tickers.csv")
    args.expert_cache_path = str(SMARTFOLIO_DIR / "dataset_default" / "expert_cache")
    args.finrag_weights_path = None
    args.finrag_prior = None
    
    # Promotion criteria
    args.promotion_min_sharpe = 0.5
    args.promotion_max_drawdown = 0.2
    
    # Resume settings
    args.resume_model_path = None
    args.reward_net_path = None
    
    # Streaming settings
    args.stream = None  # Will be set by caller if using streaming
    
    # Input dimension (will be auto-detected)
    args.input_dim = 6
    
    return args


def _run_monthly_finetune(saved_csv_path: str, year_month: str):
    """
    Run monthly fine-tuning in the current thread.
    This function is called from a separate thread.
    
    Args:
        saved_csv_path: Path to the saved monthly CSV file
        year_month: The year-month string (e.g., "2024-12")
    """
    try:
        logger.info(f"Starting monthly fine-tune for {year_month} using data from {saved_csv_path}")
        
        # Change to SmartFolio directory for imports
        original_cwd = os.getcwd()
        os.chdir(str(SMARTFOLIO_DIR))
        
        # Add SmartFolio to path
        if str(SMARTFOLIO_DIR) not in sys.path:
            sys.path.insert(0, str(SMARTFOLIO_DIR))
        
        # Import required modules from SmartFolio
        from main import fine_tune_month
        from utils.risk_profile import build_risk_profile
        from streaming.shared.locks import StreamingLocks
        import torch
        import pickle
        
        # Build default args
        args = _build_default_args()
        
        # Build risk profile
        args.risk_profile = build_risk_profile(args.risk_score)
        
        # Auto-detect num_stocks from existing pkl files
        data_dir = SMARTFOLIO_DIR / "dataset_default" / f"data_train_predict_{args.market}" / f"{args.horizon}_{args.relation_type}"
        sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')] if data_dir.exists() else []
        
        if sample_files:
            sample_path = data_dir / sample_files[0]
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)
            args.num_stocks = sample_data['features'].shape[0]
            logger.info(f"Auto-detected num_stocks: {args.num_stocks}")
        else:
            logger.warning("No existing pkl files found, cannot determine num_stocks")
            os.chdir(original_cwd)
            return
        
        # Set resume model path if baseline exists
        if os.path.exists(args.baseline_checkpoint):
            args.resume_model_path = args.baseline_checkpoint
            logger.info(f"Using baseline checkpoint: {args.resume_model_path}")
        
        # Load replay buffer if available
        replay_buffer = []
        buffer_path = os.path.join(args.save_dir, f"replay_buffer_{args.market}.pkl")
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                replay_buffer = pickle.load(f)
            logger.info(f"Loaded replay buffer with {len(replay_buffer)} samples")
        
        # Get the streaming lock for CSV access
        try:
            stream_lock = StreamingLocks().csv_write_lock
        except Exception as e:
            logger.warning(f"Could not get streaming lock: {e}, proceeding without lock")
            stream_lock = None
        
        # Run fine-tuning with stream parameter for reading from streaming CSV
        checkpoint, new_samples = fine_tune_month(
            args,
            replay_buffer=replay_buffer,
            fetch_new_data=True,  # This will fetch data and build pkl files
            stream=stream_lock,   # Pass the lock for streaming CSV access
        )
        
        logger.info(f"Monthly fine-tuning complete. Checkpoint: {checkpoint}")
        
        # Update replay buffer
        if new_samples:
            replay_buffer.extend(new_samples)
            max_buffer = args.ptr_memory_size
            if len(replay_buffer) > max_buffer:
                replay_buffer = replay_buffer[-max_buffer:]
            with open(buffer_path, "wb") as f:
                pickle.dump(replay_buffer, f)
            logger.info(f"Persisted replay buffer ({len(replay_buffer)} samples) to {buffer_path}")
        
        os.chdir(original_cwd)
        
    except Exception as e:
        logger.error(f"Error during monthly fine-tuning: {e}", exc_info=True)
        try:
            os.chdir(original_cwd)
        except:
            pass


def trigger_finetune_async(saved_csv_path: str, year_month: str):
    """
    Trigger monthly fine-tuning in an independent thread.
    Only one fine-tuning can run at a time.
    """
    global _finetune_thread
    
    with _finetune_lock:
        # Check if a fine-tuning is already running
        if _finetune_thread is not None and _finetune_thread.is_alive():
            logger.warning(f"Fine-tuning already in progress, skipping trigger for {year_month}")
            return False
        
        # Start new fine-tuning thread
        _finetune_thread = threading.Thread(
            target=_run_monthly_finetune,
            args=(saved_csv_path, year_month),
            name=f"finetune-{year_month}",
            daemon=True,
        )
        _finetune_thread.start()
        logger.info(f"Started fine-tuning thread for {year_month}")
        return True


@pw.udf
def extract_date(data_str: str) -> str:
    """Extract date from stock data JSON string."""
    try:
        data = json.loads(data_str)
        return data.get("date", "")
    except:
        return ""


@pw.udf
def extract_year_month(date_str: str) -> str:
    """Extract year-month (YYYY-MM) from date string."""
    try:
        if not date_str:
            return ""
        # Handle various date formats
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m")
            except ValueError:
                continue
        return ""
    except:
        return ""


@pw.udf
def process_stock_row(date: str, data_str: str) -> str:
    """
    Process a stock data row and buffer it.
    Returns status message.
    """
    try:
        # Parse the data
        data = json.loads(data_str)
        
        # Get shared state
        state = SharedState()
        buffer = state.stock_buffer
        
        # Add to buffer
        month_changed, saved_path = buffer.add_row(date, data)
        
        if month_changed and saved_path:
            # Trigger fine-tuning with the saved month's data
            # Extract year_month from saved_path (e.g., "/path/to/2024-12.csv" -> "2024-12")
            saved_filename = os.path.basename(saved_path)  # "2024-12.csv"
            saved_year_month = saved_filename.replace(".csv", "")  # "2024-12"
            
            # Trigger fine-tuning in an independent thread
            trigger_finetune_async(saved_path, saved_year_month)
            
            return f"Month changed. Saved to {saved_path}. Fine-tune triggered for {saved_year_month}."
        
        return f"Buffered data for {date}"
        
    except Exception as e:
        logger.error(f"Error processing stock row: {e}")
        return f"Error: {str(e)}"


class StockConsumer:
    """
    Pathway-based Kafka consumer for stock OHLCV data.
    
    Consumes from stock_data topic, detects month changes,
    buffers data, and triggers fine-tuning on month boundaries.
    """
    
    def __init__(
        self,
        broker: str = KAFKA_BROKER_STOCK,
        topic: str = STOCK_DATA_TOPIC,
        group_id: str = KAFKA_GROUP_STOCK,
    ):
        self.broker = broker
        self.topic = topic
        self.group_id = group_id
        self._state = SharedState()
    
    def build_pipeline(self) -> pw.Table:
        """
        Build the Pathway pipeline for stock data consumption.
        
        Returns:
            Pathway Table with processed stock data
        """
        # Kafka input connector
        input_table = pw_kafka.read(
            rdkafka_settings={
                "bootstrap.servers": self.broker,
                "group.id": self.group_id,
                "auto.offset.reset": "earliest",
            },
            topic=self.topic,
            format="json",
            schema=StockDataSchema,
        )
        
        # Process each row
        processed = input_table.select(
            date=pw.this.date,
            year_month=extract_year_month(pw.this.date),
            data=pw.this.data,
            status=process_stock_row(pw.this.date, pw.this.data),
        )
        
        return processed
    
    def run(self):
        """Start the Pathway runtime for stock consumption."""
        logger.info(f"Starting Stock Consumer on {self.topic}")
        
        # Build pipeline
        processed = self.build_pipeline()
        
        # Optional: Output to console for debugging
        pw.io.null.write(processed)  # Suppress output, side effects handled in UDF
        
        # Run Pathway
        pw.run(monitoring_level=pw.MonitoringLevel.NONE)


def create_stock_consumer_pipeline() -> pw.Table:
    """
    Create and return the stock consumer pipeline.
    
    This is useful for integration with other Pathway pipelines.
    """
    consumer = StockConsumer()
    return consumer.build_pipeline()


def run_stock_consumer():
    """Main entry point for stock consumer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    consumer = StockConsumer()
    consumer.run()


if __name__ == "__main__":
    run_stock_consumer()
