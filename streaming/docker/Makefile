# =============================================================================
# Makefile for SmartFolio Streaming Pipeline Docker Setup
# =============================================================================

.PHONY: help up down build logs logs-producer logs-consumer logs-kafka status clean restart

DOCKER_COMPOSE = docker compose
COMPOSE_FILE = docker-compose.yml

# Default target
help:
	@echo "SmartFolio Streaming Pipeline - Docker Commands"
	@echo "================================================"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  up              Start all containers (zookeeper → kafka-stock → kafka-user → producer → consumer)"
	@echo "  down            Stop and remove all containers"
	@echo "  build           Build Docker images"
	@echo "  rebuild         Rebuild Docker images (no cache)"
	@echo "  logs            Tail logs from all containers"
	@echo "  logs-producer   Tail producer logs only"
	@echo "  logs-consumer   Tail consumer logs only"
	@echo "  logs-kafka      Tail Kafka broker logs"
	@echo "  status          Show container status"
	@echo "  clean           Remove containers, volumes, and images"
	@echo "  restart         Restart all containers"
	@echo "  shell-producer  Open shell in producer container"
	@echo "  shell-consumer  Open shell in consumer container"
	@echo ""
	@echo "Infrastructure:"
	@echo "  kafka-up        Start only Kafka infrastructure (zookeeper + 2 brokers)"
	@echo "  kafka-down      Stop Kafka infrastructure"
	@echo "  topics          List Kafka topics"
	@echo "  create-topics   Create required Kafka topics"
	@echo ""

# =============================================================================
# Main Commands
# =============================================================================

up:
	@echo "Starting streaming pipeline..."
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d
	@echo ""
	@echo "Pipeline started! Use 'make logs' to view logs."
	@echo ""
	@$(MAKE) status

down:
	@echo "Stopping streaming pipeline..."
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down
	@echo "Pipeline stopped."

build:
	@echo "Building Docker images..."
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build

rebuild:
	@echo "Rebuilding Docker images (no cache)..."
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build --no-cache

restart: down up

# =============================================================================
# Logs
# =============================================================================

logs:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f

logs-producer:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f producer

logs-consumer:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f consumer

logs-kafka:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f kafka-stock kafka-user

logs-zookeeper:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f zookeeper

# =============================================================================
# Status & Info
# =============================================================================

status:
	@echo ""
	@echo "Container Status:"
	@echo "-----------------"
	@docker ps --filter "name=streaming-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""

ps:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) ps

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning up containers, volumes, and images..."
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down -v --rmi local
	@docker volume prune -f
	@echo "Cleanup complete."

# =============================================================================
# Shell Access
# =============================================================================

shell-producer:
	docker exec -it streaming-producer /bin/bash

shell-consumer:
	docker exec -it streaming-consumer /bin/bash

shell-kafka-stock:
	docker exec -it streaming-kafka-stock /bin/bash

shell-kafka-user:
	docker exec -it streaming-kafka-user /bin/bash

# =============================================================================
# Kafka Infrastructure Only
# =============================================================================

kafka-up:
	@echo "Starting Kafka infrastructure..."
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d zookeeper kafka-stock kafka-user
	@echo "Waiting for Kafka to be ready..."
	@sleep 10
	@$(MAKE) status

kafka-down:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) stop kafka-stock kafka-user zookeeper
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) rm -f kafka-stock kafka-user zookeeper

# =============================================================================
# Topic Management
# =============================================================================

topics:
	@echo "Stock Kafka Topics:"
	@docker exec streaming-kafka-stock kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null || echo "  (kafka-stock not running)"
	@echo ""
	@echo "User Kafka Topics:"
	@docker exec streaming-kafka-user kafka-topics --bootstrap-server localhost:9093 --list 2>/dev/null || echo "  (kafka-user not running)"

create-topics:
	@echo "Creating stock_stream topic on kafka-stock..."
	@docker exec streaming-kafka-stock kafka-topics --bootstrap-server localhost:9092 \
		--create --topic stock_stream --partitions 1 --replication-factor 1 --if-not-exists
	@echo "Creating user_stream topic on kafka-user..."
	@docker exec streaming-kafka-user kafka-topics --bootstrap-server localhost:9093 \
		--create --topic user_stream --partitions 1 --replication-factor 1 --if-not-exists
	@echo "Topics created."

# =============================================================================
# Development Helpers
# =============================================================================

# Run producer only (useful for debugging)
run-producer:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d zookeeper kafka-stock kafka-user
	@sleep 10
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up producer

# Run consumer only (useful for debugging)
run-consumer:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d zookeeper kafka-stock kafka-user
	@sleep 10
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up consumer

# Watch producer output
watch-producer:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f producer | grep -v "DEBUG"

# Watch consumer output  
watch-consumer:
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f consumer | grep -v "DEBUG"
