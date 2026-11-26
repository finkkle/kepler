# Space-grade 6-DOF Kepler Rocket Landing Simulator
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and pyproject
COPY pyproject.toml /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    pydantic \
    numpy \
    scipy \
    torch \
    stable-baselines3 \
    gymnasium

# Copy application source and checkpoints
COPY kepler /app/kepler
COPY checkpoints /app/checkpoints
COPY scripts /app/scripts
COPY README.md LICENSE /app/

# Install kepler package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Default entrypoint: launch 3D sandbox server
CMD ["python", "scripts/run_sandbox.py"]
