# Use an official python base image
FROM python:3.10

# Install prerequisites for OpenVINO, etc. (this can get quite large)
RUN apt-get update && apt-get install -y ... # necessary dependencies

# Copy your project
WORKDIR /app
COPY . /app

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port if your service is an API
EXPOSE 8000

# Command to run (for example, your Telegram bot or API)
CMD ["python", "telegram_bot.py"]