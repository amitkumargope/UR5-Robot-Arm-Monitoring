# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Download the YOLOv8 model if not present
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"

# Expose any necessary ports (if running a web interface in the future)
EXPOSE 8080

# Create necessary directories
RUN mkdir -p /app/results /app/runs

# Set environment variables
ENV PYTHONPATH=/app/src

# Define the command to run your application
CMD ["python", "-m", "src.main"]