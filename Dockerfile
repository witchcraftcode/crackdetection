FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY model.py .
COPY src ./src
COPY app.py .

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
