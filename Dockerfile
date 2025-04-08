# Use official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files to container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Railway automatically assigns a PORT
EXPOSE 8080

# Run the app
CMD ["python", "api.py"]
