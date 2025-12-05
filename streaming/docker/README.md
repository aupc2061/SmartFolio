# Docker Setup for SmartFolio Streaming Pipeline

This directory contains Docker configuration to run the streaming pipeline with **two separate Kafka brokers** - one for stock data and one for user data.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Docker Compose Network                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                                             │
│  │  Zookeeper  │◄─────────────────────────────────┐                          │
│  │   :2181     │                                  │                          │
│  └─────────────┘                                  │                          │
│         │                                         │                          │
│         ├─────────────────┬───────────────────────┘                          │
│         ▼                 ▼                                                  │
│  ┌─────────────┐   ┌─────────────┐                                           │
│  │ kafka-stock │   │ kafka-user  │                                           │
│  │   :9092     │   │   :9093     │                                           │
│  │ stock_stream│   │ user_stream │                                           │
│  └─────────────┘   └─────────────┘                                           │
│         │                 │                                                  │
│         │                 │                                                  │
│         ▼                 ▼                                                  │
│  ┌───────────────────────────────────────┐                                   │
│  │           Producer Container          │                                   │
│  │  python -m streaming.producer.run_producers                               │
│  │  • StockProducer → kafka-stock:29092  │                                   │
│  │  • UserProducer  → kafka-user:29093   │                                   │
│  └───────────────────────────────────────┘                                   │
│         │                 │                                                  │
│         ▼                 ▼                                                  │
│  ┌───────────────────────────────────────┐                                   │
│  │           Consumer Container          │                                   │
│  │  python -m streaming.consumer.run_consumers                               │
│  │  • StockConsumer ← kafka-stock:29092  │                                   │
│  │  • UserConsumer  ← kafka-user:29093   │                                   │
│  │  • FinetuneManager (12 threads)       │                                   │
│  └───────────────────────────────────────┘                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Navigate to docker directory
cd /home/mksilver30/INTER_IIT_14/streaming/docker

# Start entire pipeline
make up

# View logs
make logs

# Stop pipeline
make down
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `zookeeper` | 2181 | Kafka coordination service |
| `kafka-stock` | 9092, 29092 | Kafka broker for stock OHLCV data |
| `kafka-user` | 9093, 29093 | Kafka broker for user data |
| `producer` | - | Runs stock + user producers |
| `consumer` | - | Runs Pathway consumers + finetune manager |

## Makefile Commands

```bash
make help          # Show all available commands

# Main commands
make up            # Start all containers
make down          # Stop all containers
make build         # Build Docker images
make rebuild       # Rebuild images (no cache)
make restart       # Restart all containers

# Logs
make logs          # Tail all logs
make logs-producer # Producer logs only
make logs-consumer # Consumer logs only
make logs-kafka    # Kafka broker logs

# Status
make status        # Show container status
make topics        # List Kafka topics

# Cleanup
make clean         # Remove containers, volumes, images

# Debug
make shell-producer  # Open shell in producer container
make shell-consumer  # Open shell in consumer container
```

## Configuration

Environment variables can be set in `.env`:

```bash
# Kafka Brokers
KAFKA_BROKER_STOCK=kafka-stock:29092
KAFKA_BROKER_USER=kafka-user:29093

# Topics
STOCK_TOPIC=stock_stream
USER_TOPIC=user_stream

# Producer delays (seconds)
PRODUCER_DELAY_STOCK=0.1
PRODUCER_DELAY_USER=0.5
```

## Data Volumes

The containers mount these directories from the host:

| Host Path | Container Path | Mode |
|-----------|----------------|------|
| `../../data` | `/app/data` | read-only |
| `../../risk_artifacts` | `/app/risk_artifacts` | read-only |
| `streaming-output` (volume) | `/app/streaming/output` | read-write |

## Development

### Run just Kafka infrastructure

```bash
make kafka-up      # Start zookeeper + both kafka brokers
make kafka-down    # Stop kafka infrastructure
```

### Run individual components

```bash
make run-producer  # Start Kafka + producer only
make run-consumer  # Start Kafka + consumer only
```

### Access container shells

```bash
make shell-producer
make shell-consumer
make shell-kafka-stock
make shell-kafka-user
```

## Troubleshooting

### Kafka not starting
```bash
# Check zookeeper health
docker logs streaming-zookeeper

# Check kafka logs
docker logs streaming-kafka-stock
docker logs streaming-kafka-user
```

### Producer/Consumer not connecting
```bash
# Verify Kafka is healthy
make status

# Check topics exist
make topics

# Create topics manually if needed
make create-topics
```

### Rebuild images after code changes
```bash
make rebuild
make up
```
