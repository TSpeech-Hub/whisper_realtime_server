# Image with base support for NVIDIA CUDA
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app/src

RUN apt-get update && apt-get install -y \
    vim \
    python3 \
    python3-pip \
    git \
    libsndfile1 \
    ffmpeg \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    tzdata \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    export PATH="/root/.cargo/bin:$PATH" && \
    rustup default stable && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:$PATH"

COPY resources /app/resources

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt
RUN pip install -U openai-whisper
RUN pip install git+https://github.com/linto-ai/whisper-timestamped 

COPY src /app/src

CMD ["python3", "layer_server.py"]
