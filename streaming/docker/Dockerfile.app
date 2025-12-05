# =============================================================================
# Dockerfile for SmartFolio Streaming Pipeline
# Used for both Producer and Consumer containers
# =============================================================================

FROM python:3.10-slim

# Install system dependencies (including OpenGL libs for OpenCV and poppler for pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for pathway and some dependencies)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    kafka-python \
    "pathway[xpack-llm-docs]" \
    pandas \
    numpy \
    torch \
    torchvision \
    scikit-learn \
    stable-baselines3 \
    torch-geometric \
    yfinance \
    pillow \
    requests \
    gym \
    gymnasium \
    # KYC Document Processing
    paddleocr \
    paddlepaddle \
    pdf2image \
    docling \
    # Video verification
    opencv-python \
    facenet-pytorch \
    # FastAPI for KYC API
    fastapi \
    uvicorn \
    python-multipart \
    pydantic

# Install additional requirements if they exist
RUN if [ -f /app/requirements.txt ]; then pip install --no-cache-dir -r /app/requirements.txt || true; fi

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command (overridden by docker-compose)
CMD ["python", "-m", "streaming.run_all"]
