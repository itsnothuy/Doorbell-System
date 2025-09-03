# Doorbell Security System - Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for face recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-web.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/known_faces data/blacklist_faces data/captures data/logs templates

# Set environment variables
ENV DEVELOPMENT_MODE=true
ENV PORT=5000
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# Run the application
CMD ["python", "app.py"]
