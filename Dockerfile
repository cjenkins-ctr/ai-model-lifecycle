FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy application code
COPY ./app ./app

# Create registry directory
RUN mkdir -p /app/registry

# Expose API port
EXPOSE 8000

# Run the API server
CMD ["model-serve"]
