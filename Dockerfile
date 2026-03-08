# Multi-stage Dockerfile for Insurance Claims Fraud Detection System
# Optimized for production deployment

# Stage 1: Builder
FROM python:3.9-slim as builder

LABEL maintainer="Data Science Team"
LABEL description="Enterprise Insurance Claims Fraud Detection System"
LABEL version="2.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY insurance_claims_analysis.py .
COPY config.py ./config.py 2>/dev/null || true

# Create necessary directories
RUN mkdir -p \
    /app/models \
    /app/data \
    /app/logs \
    /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models \
    DATA_PATH=/app/data \
    OUTPUT_PATH=/app/output \
    LOG_LEVEL=INFO

# Expose port for API
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import insurance_claims_analysis; print('OK')" || exit 1

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command
CMD ["python", "insurance_claims_analysis.py"]
