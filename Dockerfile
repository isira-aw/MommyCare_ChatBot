# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Install necessary system dependencies (e.g., for SpeechRecognition and OpenVINO if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for the FastAPI application
EXPOSE 8080

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
