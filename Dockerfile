# Dockerfile for ASTRA VISION (Defence Object Recognition System)
# CPU-first container deployment running Streamlit on port 8501

FROM python:3.11-slim

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-specific PyTorch wheels to minimize image footprint
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY config.yaml .
COPY .env.example .env
COPY data/ ./data/
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY assets/ ./assets/

# NOTE: Model weights are generated locally via 'make train' and 'make probe'
# Mount local models directory when running the container:
#   docker run -p 8501:8501 -v $(pwd)/models:/app/models astra-vision:latest

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
