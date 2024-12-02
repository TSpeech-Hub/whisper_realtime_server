# Usa un'immagine base con Python e supporto per NVIDIA CUDA
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Imposta l'area geografica per evitare richieste interattive
ENV DEBIAN_FRONTEND=noninteractive

# Imposta la directory di lavoro all'interno del container
WORKDIR /app/src

# Aggiorna i pacchetti, installa Rust e le dipendenze di base
RUN apt-get update && apt-get install -y \
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

# Aggiorna il PATH per includere Rust (per ogni step successivo nel Dockerfile)
ENV PATH="/root/.cargo/bin:$PATH"

# Copia i file di configurazione (certificati SSL) e la cartella resources nella directory appropriata
COPY resources /app/resources

# Copia i requisiti Python e installali
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copia il codice dell'applicazione nella cartella src
COPY src /app/src

# Espone le porte utilizzate dall'applicazione
EXPOSE 8000 8001 8002

# Comando di avvio del server
CMD ["python3", "layer_server.py"]
