FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyTorch and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Check if requirements.txt exists, if not create a basic one
# Copy requirements if they exist
COPY requirements.txt* ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pandas \
    torch \
    torch-geometric \
    stable-baselines3 \
    scikit-learn \
    numpy \
    && if [ -f requirements.txt ]; then pip install --default-timeout=1000 --no-cache-dir -r requirements.txt; fi

# Copy the entire SmartFolio application
COPY . .

# Expose the API port
EXPOSE 8000

# Run the SmartFolio API server
WORKDIR /app
CMD ["python", "api/server.py", "--host", "0.0.0.0", "--port", "8000"]

