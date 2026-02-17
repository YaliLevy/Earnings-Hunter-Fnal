# Dockerfile for The Earnings Hunter
# Multi-stage build: React frontend + FastAPI backend

# ========== Stage 1: Build React Frontend ==========
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# Install dependencies first (better caching)
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --silent

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# ========== Stage 2: Python Backend + Serve Static ==========
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data for sentiment analysis
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# Download TextBlob corpora
RUN python -m textblob.download_corpora

# Copy application code
COPY . .

# Copy built frontend from Stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Create data directories
RUN mkdir -p data/raw data/processed data/models data/cache data/training

# Railway uses PORT env var
EXPOSE ${PORT:-8000}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start FastAPI server
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
