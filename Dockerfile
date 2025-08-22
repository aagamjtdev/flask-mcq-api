# Use a lightweight Python image
FROM python:3.10-slim

# Install system dependencies (Tesseract OCR + fonts)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose port (Render will use this)
EXPOSE 5000

# Start app with Gunicorn (better for production than python app.py)
# "app:app" means app.py file and Flask app = app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

