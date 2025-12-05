"""
Configuration constants for the streaming pipeline.
Supports both local development and Docker deployment via environment variables.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Helper to detect SmartFolio root directory
# =============================================================================
def _detect_smartfolio_root() -> Path:
    """
    Detect the SmartFolio root directory by walking up from this file.
    The streaming folder is expected to be at SmartFolio/streaming/
    """
    current = Path(__file__).resolve().parent  # streaming/
    parent = current.parent  # SmartFolio/
    if (parent / "main.py").exists():
        return parent
    # Fallback to environment variable or current directory
    return Path(os.getenv("SMARTFOLIO_ROOT", str(parent)))


class Config:
    """
    Configuration class that supports environment variable overrides.
    Use Config.ATTR_NAME to access configuration values.
    """
    
    # =============================================================================
    # Base Paths (can be overridden via environment variables)
    # =============================================================================
    # SmartFolio root is auto-detected or set via environment
    SMARTFOLIO_DIR = Path(os.getenv("SMARTFOLIO_ROOT", str(_detect_smartfolio_root())))
    BASE_DIR = Path(os.getenv("BASE_DIR", str(SMARTFOLIO_DIR.parent)))
    DATA_DIR = Path(os.getenv("DATA_DIR", str(SMARTFOLIO_DIR / "data")))
    STREAMING_DIR = Path(os.getenv("STREAMING_DIR", str(SMARTFOLIO_DIR / "streaming")))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(STREAMING_DIR / "output")))
    
    # Risk artifacts path (pre-trained model for risk scoring)
    RISK_ARTIFACTS = Path(os.getenv("RISK_ARTIFACTS", str(SMARTFOLIO_DIR / "risk_artifacts")))
    RISK_ARTIFACTS_DIR = RISK_ARTIFACTS  # Alias
    
    # =============================================================================
    # Input Data Paths
    # =============================================================================
    OHLCV_RAW_PATH = Path(os.getenv("OHLCV_RAW_PATH", str(DATA_DIR / "ohlcv_raw.csv")))
    STOCK_DATA_PATH = OHLCV_RAW_PATH  # Alias
    INPUT_CSV_PATH = Path(os.getenv("INPUT_CSV_PATH", str(DATA_DIR / "input.csv")))
    USER_DATA_PATH = INPUT_CSV_PATH  # Alias
    TICKERS_PATH = Path(os.getenv("TICKERS_PATH", str(SMARTFOLIO_DIR / "tickers.csv")))
    TEST_IMAGES_DIR = DATA_DIR / "test_images"


# =============================================================================
# Backward compatibility - expose as module-level constants
# =============================================================================
BASE_DIR = Config.BASE_DIR
DATA_DIR = Config.DATA_DIR
SMARTFOLIO_DIR = Config.SMARTFOLIO_DIR
STREAMING_DIR = Config.STREAMING_DIR
OUTPUT_DIR = Config.OUTPUT_DIR

# Risk artifacts path (pre-trained model for risk scoring)
RISK_ARTIFACTS_DIR = Config.RISK_ARTIFACTS_DIR

# =============================================================================
# Input Data Paths
# =============================================================================
STOCK_DATA_PATH = Config.STOCK_DATA_PATH
OHLCV_RAW_PATH = Config.OHLCV_RAW_PATH
USER_DATA_PATH = Config.USER_DATA_PATH
INPUT_CSV_PATH = Config.INPUT_CSV_PATH
TICKERS_PATH = Config.TICKERS_PATH
TEST_IMAGES_DIR = Config.TEST_IMAGES_DIR

# =============================================================================
# Output Paths
# =============================================================================
MONTHLY_STOCK_DATA_DIR = OUTPUT_DIR / "monthly_stock_data"
MONTHLY_DATA_DIR = MONTHLY_STOCK_DATA_DIR  # Alias for backward compatibility
USER_DATABASE_DIR = OUTPUT_DIR / "user_database"
USER_RISK_DATA_DIR = USER_DATABASE_DIR  # Alias for backward compatibility
MODELS_DIR = OUTPUT_DIR / "models"
FINETUNE_CHECKPOINTS_DIR = MODELS_DIR / "finetune_checkpoints"
DEBUG_DIR = OUTPUT_DIR / "debug"

# =============================================================================
# Kafka Configuration (Docker-aware) - Two Separate Brokers
# =============================================================================
# Stock Kafka Broker (for stock/OHLCV data)
KAFKA_BROKER_STOCK = os.getenv("KAFKA_BROKER_STOCK", "localhost:9092")
# User Kafka Broker (for user data)
KAFKA_BROKER_USER = os.getenv("KAFKA_BROKER_USER", "localhost:9093")

# Legacy single broker (for backward compatibility - defaults to stock broker)
KAFKA_BROKER = os.getenv("KAFKA_BROKER", KAFKA_BROKER_STOCK)

KAFKA_GROUP_ID_STOCK = os.getenv("KAFKA_GROUP_STOCK", "stock-consumer-group")
KAFKA_GROUP_ID_USER = os.getenv("KAFKA_GROUP_USER", "user-consumer-group")

# Kafka Topics
STOCK_DATA_TOPIC = os.getenv("STOCK_TOPIC", "stock_stream")
USER_DATA_TOPIC = os.getenv("USER_TOPIC", "user_stream")
PROCESSED_USER_TOPIC = "processed_user_data"

# Kafka Group IDs
KAFKA_GROUP_STOCK = KAFKA_GROUP_ID_STOCK  # Alias
KAFKA_GROUP_USER = KAFKA_GROUP_ID_USER  # Alias

# Kafka rdkafka settings for Pathway
def get_rdkafka_settings(group_id: str, broker: Optional[str] = None) -> Dict[str, str]:
    """Get rdkafka settings for Pathway Kafka connector.
    
    Args:
        group_id: Consumer group ID
        broker: Kafka broker address. If None, uses KAFKA_BROKER_STOCK.
    """
    return {
        "bootstrap.servers": broker or KAFKA_BROKER_STOCK,
        "group.id": group_id,
        "session.timeout.ms": "6000",
        "auto.offset.reset": "earliest",
    }


def get_stock_rdkafka_settings() -> Dict[str, str]:
    """Get rdkafka settings for stock data consumer."""
    return get_rdkafka_settings(KAFKA_GROUP_STOCK, KAFKA_BROKER_STOCK)


def get_user_rdkafka_settings() -> Dict[str, str]:
    """Get rdkafka settings for user data consumer."""
    return get_rdkafka_settings(KAFKA_GROUP_USER, KAFKA_BROKER_USER)

# =============================================================================
# Streaming Configuration (from environment or defaults)
# =============================================================================
# Delay between sending stock data rows (seconds)
STOCK_DATA_DELAY = float(os.getenv("PRODUCER_DELAY_STOCK", "20"))
PRODUCER_DELAY_STOCK = STOCK_DATA_DELAY  # Alias

# Delay between sending user data records (seconds)
USER_DATA_DELAY = float(os.getenv("PRODUCER_DELAY_USER", "200"))
PRODUCER_DELAY_USER = USER_DATA_DELAY  # Alias

# How often finetune threads check for work (seconds)
FINETUNE_POLL_INTERVAL = 5.0

# =============================================================================
# Risk Group Configuration (12 groups)
# =============================================================================
NUM_RISK_GROUPS = 12

# Risk score boundaries for each group
# Group i covers [RISK_BOUNDARIES[i][0], RISK_BOUNDARIES[i][1])
RISK_BOUNDARIES: List[Tuple[float, float]] = [
    (0.0, 8.33),      # Group 0: Very Conservative
    (8.33, 16.67),    # Group 1: Conservative
    (16.67, 25.0),    # Group 2: Conservative-Moderate
    (25.0, 33.33),    # Group 3: Low-Moderate
    (33.33, 41.67),   # Group 4: Moderate-Low
    (41.67, 50.0),    # Group 5: Moderate
    (50.0, 58.33),    # Group 6: Moderate-High
    (58.33, 66.67),   # Group 7: High-Moderate
    (66.67, 75.0),    # Group 8: Moderately Aggressive
    (75.0, 83.33),    # Group 9: Aggressive
    (83.33, 91.67),   # Group 10: Very Aggressive
    (91.67, 100.01),  # Group 11: Extremely Aggressive (100.01 to include 100)
]

RISK_GROUP_NAMES: List[str] = [
    "Very Conservative",
    "Conservative", 
    "Conservative-Moderate",
    "Low-Moderate",
    "Moderate-Low",
    "Moderate",
    "Moderate-High",
    "High-Moderate",
    "Moderately Aggressive",
    "Aggressive",
    "Very Aggressive",
    "Extremely Aggressive",
]

def get_risk_group(risk_score: float) -> int:
    """
    Get the risk group index (0-11) for a given risk score (0-100).
    """
    for i, (low, high) in enumerate(RISK_BOUNDARIES):
        if low <= risk_score < high:
            return i
    # Edge case: if score is exactly 100 or above
    return NUM_RISK_GROUPS - 1


def user_risk_to_portfolio_risk(user_risk_score: float) -> float:
    """
    Convert user risk score (0-100) to portfolio risk score (0.0-1.0).
    
    The user risk score comes from the investor risk profiling system (KYC, questionnaire).
    The portfolio risk score is used for training the SmartFolio model.
    
    Args:
        user_risk_score: User risk score from 0 to 100
        
    Returns:
        Portfolio risk score from 0.0 to 1.0
        
    Examples:
        user_risk_to_portfolio_risk(50) -> 0.5
        user_risk_to_portfolio_risk(10) -> 0.1
        user_risk_to_portfolio_risk(100) -> 1.0
    """
    return max(0.0, min(1.0, user_risk_score / 100.0))


def portfolio_risk_to_user_risk(portfolio_risk_score: float) -> float:
    """
    Convert portfolio risk score (0.0-1.0) to user risk score (0-100).
    
    Args:
        portfolio_risk_score: Portfolio risk score from 0.0 to 1.0
        
    Returns:
        User risk score from 0 to 100
    """
    return max(0.0, min(100.0, portfolio_risk_score * 100.0))


def get_model_path(risk_group: int) -> Path:
    """Get the model checkpoint path for a risk group."""
    return MODELS_DIR / f"model_risk_{risk_group}.zip"

def get_manifest_path(risk_group: int) -> Path:
    """Get the manifest path for a risk group."""
    return MODELS_DIR / f"manifest_risk_{risk_group}.json"

# =============================================================================
# Fine-tuning Configuration
# =============================================================================
FINETUNE_STEPS = 5000

# SmartFolio training parameters
SMARTFOLIO_MARKET = "custom"
SMARTFOLIO_HORIZON = 1
SMARTFOLIO_RELATION_TYPE = "hy"


def get_risk_score_dir(base_dir: str, risk_score: float) -> str:
    """
    Get the checkpoint directory for a specific risk score.
    Matches the convention in main.py.
    
    Args:
        base_dir: Base directory (e.g., 'checkpoints')
        risk_score: Risk score 0.0-1.0
        
    Returns:
        Directory path like 'checkpoints_risk05' for risk_score=0.5
    
    Examples:
        base_dir='checkpoints', risk_score=0.5 -> 'checkpoints_risk05'
        base_dir='./checkpoints', risk_score=0.1 -> './checkpoints_risk01'
    """
    # Convert risk score to tag (0.5 -> '05', 0.1 -> '01')
    risk_tag = str(risk_score).replace('.', '')
    # Append risk tag to directory name
    return f"{str(base_dir).rstrip('/')}_risk{risk_tag}"


def get_baseline_checkpoint(risk_score: float) -> Path:
    """
    Get the baseline checkpoint path for a specific risk score.
    
    Args:
        risk_score: Risk score 0.0-1.0
        
    Returns:
        Path to baseline.zip in the appropriate risk-score directory
    """
    risk_dir = get_risk_score_dir(str(SMARTFOLIO_DIR / "checkpoints"), risk_score)
    return Path(risk_dir) / "baseline.zip"


# Default baseline (for backward compatibility) - uses risk_score=0.5
BASELINE_CHECKPOINT = get_baseline_checkpoint(0.5)

# =============================================================================
# Create output directories
# =============================================================================
def ensure_output_dirs():
    """Create all required output directories."""
    dirs = [
        OUTPUT_DIR,
        MONTHLY_STOCK_DATA_DIR,
        USER_DATABASE_DIR,
        MODELS_DIR,
        FINETUNE_CHECKPOINTS_DIR,
        DEBUG_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# Ensure directories exist on import
ensure_output_dirs()
