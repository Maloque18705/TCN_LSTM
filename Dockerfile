# Lightweight Dockerfile for TCN_LSTM project (CPU)
# Based on Python 3.10 slim. Installs system deps required by numpy/scipy/pandas

FROM python:3.10-slim

# Reduce Python output noise and bytecode files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libffi-dev \
    libssl-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Create non-root user and switch
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default command: run main.py
CMD ["python", "main.py"]

# Notes:
# - For GPU/TensorFlow with CUDA, use an official tensorflow/tensorflow:2.10.1-gpu base image
#   or adjust the Dockerfile to install CUDA drivers; this file is CPU-only.