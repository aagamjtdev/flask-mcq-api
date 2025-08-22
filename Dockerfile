# Use official Python base image
FROM python:3.10-slim

# Install system dependencies (Tesseract OCR and libraries)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code into the container
COPY . .

# Expose port (Render uses 5000 by default)
EXPOSE 5000

# Start the app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
