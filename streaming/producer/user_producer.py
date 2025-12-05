"""
User Data Producer

Reads user data from input.csv with document paths,
encodes documents (Aadhaar, PAN, ITR, Video) to base64,
and streams to Kafka user_data topic.
"""

import json
import time
import logging
import pandas as pd
from pathlib import Path
from typing import Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streaming.config import (
    KAFKA_BROKER_USER,
    USER_DATA_TOPIC,
    INPUT_CSV_PATH,
    PRODUCER_DELAY_USER,
    DATA_DIR,
)
from streaming.shared.utils import encode_image_base64, encode_video_base64, prepare_user_message

logger = logging.getLogger(__name__)


class UserProducer:
    """
    Kafka producer for user data with document encoding.
    
    Reads from input.csv and streams each user as a JSON message
    to the user_data Kafka topic, with documents (Aadhaar, PAN, ITR, Video) 
    encoded as base64.
    """
    
    def __init__(
        self,
        broker: str = KAFKA_BROKER_USER,
        topic: str = USER_DATA_TOPIC,
        delay: float = PRODUCER_DELAY_USER,
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
                max_request_size=104857600,  # 100MB for large video payloads
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
            future.get(timeout=60)  # Longer timeout for large video payloads
            logger.info(f"Sent user data for key={key}")
            return True
        except KafkaError as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def load_users(self, csv_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """Load user data from CSV."""
        csv_path = csv_path or INPUT_CSV_PATH
        
        if not csv_path.exists():
            logger.error(f"User data file not found: {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} users from {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load user data: {e}")
            return None
    
    def stream_data(self, csv_path: Optional[Path] = None, loop: bool = False):
        """
        Stream user data from CSV to Kafka.
        
        Args:
            csv_path: Path to input.csv (uses default if None)
            loop: If True, restart from beginning after reaching end
        """
        csv_path = csv_path or INPUT_CSV_PATH
        
        if not self.producer:
            if not self.connect():
                return
        
        self._running = True
        logger.info(f"Starting user data stream from {csv_path}")
        
        try:
            while self._running:
                df = self.load_users(csv_path)
                
                if df is None or df.empty:
                    logger.error("No user data to stream")
                    break
                
                # Stream each user
                for idx, row in df.iterrows():
                    if not self._running:
                        break
                    
                    # Prepare message with encoded documents (Aadhaar, PAN, ITR, Video)
                    message = prepare_user_message(row, DATA_DIR)
                    
                    if message:
                        # Use user id as key for partitioning
                        user_id = str(message.get("userid", idx))
                        
                        # Log document encoding status
                        docs_encoded = {
                            "aadhar": bool(message.get("aadhar_base64")),
                            "pan": bool(message.get("pan_base64")),
                            "itr": bool(message.get("itr_base64")),
                            "video": bool(message.get("video_base64")),
                        }
                        logger.info(f"User {user_id} documents encoded: {docs_encoded}")
                        
                        if self.send_message(user_id, message):
                            logger.info(f"Sent user data for user_id={user_id}")
                        else:
                            logger.warning(f"Failed to send data for user_id={user_id}")
                        
                        # Delay between users
                        time.sleep(self.delay)
                
                if not loop:
                    break
                    
                logger.info("Restarting user stream from beginning...")
                time.sleep(5)  # Pause before restart
                
        except KeyboardInterrupt:
            logger.info("User producer interrupted by user")
        except Exception as e:
            logger.exception(f"Error in user stream: {e}")
        finally:
            self._running = False
            logger.info("User data stream ended")
    
    def stop(self):
        """Stop the streaming loop."""
        self._running = False


def run_user_producer(loop: bool = False, delay: Optional[float] = None):
    """
    Main entry point for user producer.
    
    Args:
        loop: Whether to loop through data continuously
        delay: Override default delay between messages
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    producer = UserProducer()
    if delay is not None:
        producer.delay = delay
    
    try:
        producer.stream_data(loop=loop)
    finally:
        producer.disconnect()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="User Data Kafka Producer")
    parser.add_argument("--loop", action="store_true", help="Loop through data continuously")
    parser.add_argument(
        "--delay",
        type=float,
        default=PRODUCER_DELAY_USER,
        help=f"Delay between user messages in seconds (default: {PRODUCER_DELAY_USER})",
    )
    
    args = parser.parse_args()
    run_user_producer(loop=args.loop, delay=args.delay)
