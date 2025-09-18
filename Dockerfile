# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# add installation of uv
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install uv

# Install the required Python packages
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY src/ ./src

# Set Python path to include src directory
ENV PYTHONPATH=/app/src

# Expose the port your FastMCP server listens on (e.g., 8000)
EXPOSE 8000

# Define the command to run your FastMCP server
CMD ["python", "src/ml_server.py", "--transport", "http"]