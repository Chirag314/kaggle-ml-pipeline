# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Run pipeline by default
CMD ["python", "src/main.py"]
