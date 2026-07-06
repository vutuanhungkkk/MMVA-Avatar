# Base image: NVIDIA CUDA 12.6 + cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

# Build arguments and environment variables
ARG PYTHON_VERSION=3.11
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    ffmpeg \
    portaudio19-dev \
    libsndfile1 \
    libsndfile1-dev \
    libgomp1 \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

WORKDIR /app

# Upgrade pip core tools
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.6
RUN pip install --no-cache-dir \
    torch==2.6.0+cu126 \
    torchvision==0.21.0+cu126 \
    torchaudio==2.6.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# Clean pre-installed distutils packages to prevent pip conflicts
RUN apt-get remove -y \
    python3-blinker \
    python3-distro \
    2>/dev/null || true

# Install project requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY README.md .
COPY start.sh .

# Setup permissions and mount points
RUN chmod +x start.sh && \
    mkdir -p models documents vector_db

# Expose required ports
EXPOSE 8000 8080

# Health check to ensure the API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Start the application
CMD ["/bin/bash", "start.sh"]