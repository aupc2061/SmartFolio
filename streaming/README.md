# Streaming Pipeline for SmartFolio Portfolio Management

A Kafka-Pathway based streaming system for real-time portfolio management with risk-adaptive fine-tuning.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STREAMING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────┐                                     │
│  │           PRODUCERS                 │                                     │
│  │  ┌──────────────────────────────┐   │                                     │
│  │  │     Stock Producer           │   │  Reads: data/ohlcv_raw.csv          │
│  │  │  (Thread 1 - stock_data)     │───┼───▶ Kafka Topic: stock_data         │
│  │  └──────────────────────────────┘   │                                     │
│  │  ┌──────────────────────────────┐   │                                     │
│  │  │     User Producer            │   │  Reads: data/input.csv              │
│  │  │  (Thread 2 - user_data)      │───┼───▶ Kafka Topic: user_data          │
│  │  └──────────────────────────────┘   │  (images encoded as base64)         │
│  └─────────────────────────────────────┘                                     │
│                     │                                                        │
│                     ▼                                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         KAFKA BROKER                                   │  │
│  │                      (localhost:9092)                                  │  │
│  │   Topics: stock_data, user_data                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                     │                                                        │
│                     ▼                                                        │
│  ┌─────────────────────────────────────┐                                     │
│  │          CONSUMERS (Pathway)        │                                     │
│  │  ┌──────────────────────────────┐   │                                     │
│  │  │   Stock Consumer             │   │  • Buffers stock data               │
│  │  │   (pw.io.kafka.read)         │   │  • Detects month changes            │
│  │  │   @pw.udf processing         │───┼───▶ Saves monthly CSV               │
│  │  └──────────────────────────────┘   │  • Triggers fine-tuning             │
│  │  ┌──────────────────────────────┐   │                                     │
│  │  │   User Consumer              │   │  • Calculates risk scores           │
│  │  │   (pw.io.kafka.read)         │   │  • Assigns to 12 risk groups        │
│  │  │   @pw.udf with RiskScorer    │───┼───▶ Persists user data              │
│  │  └──────────────────────────────┘   │                                     │
│  └─────────────────────────────────────┘                                     │
│                     │                                                        │
│                     ▼                                                        │
│  ┌─────────────────────────────────────┐                                     │
│  │      FINETUNE MANAGER               │                                     │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐       │                                     │
│  │  │ G0 │ │ G1 │ │ G2 │ │... │  12 Threads (one per risk group)            │
│  │  └────┘ └────┘ └────┘ └────┘       │                                     │
│  │  Each thread fine-tunes PPO model   │                                     │
│  │  when triggered by month change     │───▶ SmartFolio Checkpoints          │
│  └─────────────────────────────────────┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Producers (`streaming/producer/`)
- **StockProducer**: Reads OHLCV data from `data/ohlcv_raw.csv` and streams to Kafka
- **UserProducer**: Reads user data from `data/input.csv`, encodes images to base64, streams to Kafka

### 2. Consumers (`streaming/consumer/`)
- **StockConsumer**: Pathway-based consumer using `pw.io.kafka.read` and `@pw.udf`
  - Buffers incoming stock data
  - Detects month boundaries
  - Saves monthly CSV files
  - Triggers fine-tuning on month change
  
- **UserConsumer**: Pathway-based consumer with risk scoring
  - Uses `OnlineRiskScorer` from `investor_risk_scorer.py`
  - Calculates risk scores (0-100)
  - Assigns users to one of 12 risk groups
  - Persists user risk data

### 3. Fine-tune Manager (`streaming/consumer/finetune_manager.py`)
- Manages 12 worker threads (one per risk group)
- Each thread has its own trigger queue
- Fine-tunes SmartFolio PPO models using streamed data
- Uses risk-adjusted learning rates

### 4. Shared State (`streaming/shared/`)
- **locks.py**: Thread-safe locks for buffer, CSV, finetune, and model operations
- **state.py**: Shared state singleton with stock buffer and user risk data
- **utils.py**: Utilities for image encoding, stock row parsing, etc.

## 12 Risk Groups

Users are assigned to one of 12 risk groups based on their risk score:

| Group | Risk Score Range | Risk Profile |
|-------|-----------------|--------------|
| 0     | 0.00 - 8.33     | Very Conservative |
| 1     | 8.33 - 16.67    | Conservative |
| 2     | 16.67 - 25.00   | Conservative |
| 3     | 25.00 - 33.33   | Conservative-Moderate |
| 4     | 33.33 - 41.67   | Moderate |
| 5     | 41.67 - 50.00   | Moderate |
| 6     | 50.00 - 58.33   | Moderate |
| 7     | 58.33 - 66.67   | Moderate-Aggressive |
| 8     | 66.67 - 75.00   | Aggressive |
| 9     | 75.00 - 83.33   | Aggressive |
| 10    | 83.33 - 91.67   | Very Aggressive |
| 11    | 91.67 - 100.00  | Very Aggressive |

## Prerequisites

### 1. Install Dependencies

```bash
# Core dependencies
pip install kafka-python pathway pandas numpy torch

# For risk scoring
pip install scikit-learn

# For SmartFolio
pip install stable-baselines3 torch-geometric

# Optional: yfinance for data download
pip install yfinance
```

### 2. Start Kafka

```bash
# Using Docker (recommended)
docker run -d --name kafka \
  -p 9092:9092 \
  -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  confluentinc/cp-kafka:latest

# Or start local Kafka installation
kafka-server-start.sh config/server.properties
```

### 3. Create Kafka Topics

```bash
# Create topics
kafka-topics.sh --create --topic stock_data --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
kafka-topics.sh --create --topic user_data --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

### 4. Prepare Data

Ensure the following files exist:
- `data/ohlcv_raw.csv` - Stock OHLCV data (MultiIndex format)
- `data/input.csv` - User data with columns: id, first, last, image_path, age, gender, etc.
- `risk_artifacts/` - Pre-trained risk scoring model artifacts

## Usage

### Run Everything (Recommended)

```bash
# From project root
python -m streaming.run_all
```

### Run Components Separately

```bash
# Run only producers
python -m streaming.run_all --producers-only

# Run only consumers
python -m streaming.run_all --consumers-only

# Run individual components
python -m streaming.producer.run_producers
python -m streaming.consumer.run_consumers
```

### Configuration Options

```bash
# Custom delays between messages
python -m streaming.run_all --stock-delay 0.5 --user-delay 2.0

# Run data once (no loop)
python -m streaming.run_all --no-loop

# Debug logging
python -m streaming.run_all --log-level DEBUG
```

## Configuration (`streaming/config.py`)

Key settings:

```python
# Kafka
KAFKA_BROKER = "localhost:9092"
STOCK_DATA_TOPIC = "stock_data"
USER_DATA_TOPIC = "user_data"

# Producer delays (seconds)
PRODUCER_DELAY_STOCK = 0.1  # 100ms between stock rows
PRODUCER_DELAY_USER = 1.0   # 1 second between users

# Risk scoring
NUM_RISK_GROUPS = 12
RISK_ARTIFACTS_DIR = Path("/home/mksilver30/INTER_IIT_14/risk_artifacts")
```

## File Structure

```
streaming/
├── __init__.py
├── config.py                 # Configuration constants
├── run_all.py               # Main entry point
├── shared/
│   ├── __init__.py
│   ├── locks.py             # Thread-safe locks
│   ├── state.py             # Shared state management
│   └── utils.py             # Utility functions
├── producer/
│   ├── __init__.py
│   ├── stock_producer.py    # OHLCV data producer
│   ├── user_producer.py     # User data producer
│   └── run_producers.py     # Producer manager
└── consumer/
    ├── __init__.py
    ├── schemas.py           # Pathway schemas
    ├── stock_consumer.py    # Stock data consumer
    ├── user_consumer.py     # User data consumer
    ├── finetune_manager.py  # 12 fine-tuning threads
    └── run_consumers.py     # Consumer manager
```

## Integration with SmartFolio

The streaming pipeline integrates with SmartFolio through:

1. **`fine_tune_month_streaming()`** in `SmartFolio/main.py`
   - Called by FinetuneManager workers
   - Uses streamed CSV data
   - Supports risk-adjusted learning rates

2. **`fetch_ohlcv_streamed()`** in `SmartFolio/gen_data/build_dataset_yf.py`
   - Loads monthly CSV produced by streaming consumer
   - Thread-safe with optional lock parameter

## Monitoring

The pipeline logs status periodically:

```
2024-01-15 10:30:00 - Pathway: running | Users processed: 15 | Finetune workers: 12/12 active, 3 runs
```

Check individual component status:
```python
from streaming.consumer.run_consumers import ConsumerManager

manager = ConsumerManager()
status = manager.get_status()
print(status)
```

## Troubleshooting

### Kafka Connection Issues
```bash
# Check Kafka is running
kafka-broker-api-versions.sh --bootstrap-server localhost:9092
```

### Pathway Import Errors
```bash
# Ensure Pathway is installed
pip install pathway
```

### Risk Scorer Not Found
Ensure risk artifacts exist at the configured path:
```bash
ls /home/mksilver30/INTER_IIT_14/risk_artifacts/
```

## License

Same as SmartFolio project.
