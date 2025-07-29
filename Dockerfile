FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y wget

# Install Miniconda
RUN mkdir -p /root/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm /root/miniconda3/miniconda.sh && \
    echo "source activate app" >> ~/.bashrc

# Accept Terms of Service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and prepare the conda env
RUN conda create -n app python=3.12 -y && \
    conda install -n app pip -y

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install torch
RUN conda run -n app pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Python dependencies
RUN conda run -n app pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default environment variables
ENV DISABLE_CUDA=false \
    PORT=8000 \
    PYTHONPATH=/app

# Run app using conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "app", "python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
