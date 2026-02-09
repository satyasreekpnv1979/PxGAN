FROM pytorch/pytorch:2.0.1-cpu

# Set working directory
WORKDIR /workspace/pxgan

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables for reproducibility
ENV OMP_NUM_THREADS=32
ENV MKL_NUM_THREADS=32
ENV PYTHONUNBUFFERED=1

# Create directories
RUN mkdir -p /workspace/pxgan/raw_data \
    /workspace/pxgan/processed_data \
    /workspace/pxgan/experiments \
    /workspace/pxgan/checkpoints

# Expose TensorBoard port
EXPOSE 6006

# Default command
CMD ["/bin/bash"]

# Usage:
# Build: docker build -t pxgan .
# Run:   docker run -it --rm -v $(pwd)/raw_data:/workspace/pxgan/raw_data pxgan
# Train: docker run -it --rm -v $(pwd):/workspace/pxgan pxgan python scripts/train.py --config config/default.yaml --data_dir processed_data
