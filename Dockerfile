FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (kept minimal for slim image)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and assets
COPY src/ ./src/
COPY assets/ ./assets/
COPY models/ ./models/
COPY data/ ./data/

# Ensure directories exist (no-op if already there)
RUN mkdir -p data models

# Copy local trained model instead of running the training script
COPY models/ ./models/

# Expose container port
EXPOSE 8090

# Environment variables
ENV API_PORT=8090
ENV DB_PATH=data/db.nba.sqlite
ENV MODEL_PATH=models/nba_over30_model.pt

# Start the API server
CMD sh -c "uvicorn src.api.main:app --host 0.0.0.0 --port ${API_PORT:-8090}"
