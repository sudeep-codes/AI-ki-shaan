# =====================================================================
# AI-ki-shan Dockerfile
# Optimized for Hugging Face Spaces & Production Containers (16GB RAM CPU/GPU)
# =====================================================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    HOME=/home/user

# Install system dependencies for PDF parsing, image processing, and unstructured tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (UID 1000 is required for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Copy dependency definition first for optimal layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Create necessary persistent storage directories and assign permissions
RUN mkdir -p /app/data/chroma_db /app/data/uploads /app/chroma_data && \
    chown -R user:user /app /home/user

# Switch to non-root user
USER user

# Expose port (HF Spaces default is 7860)
EXPOSE 7860

# Start FastAPI backend with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
