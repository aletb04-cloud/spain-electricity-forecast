# 1. Base image: lightweight Debian Linux with Python 3.11 pre-installed
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files and force unbuffered stdout/stderr logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 3. Install Linux system runtime dependency required by LightGBM (OpenMP)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 4. Set working directory inside the container
WORKDIR /app

# 5. Copy and install Python dependencies first (optimizes Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy application source code and trained model artifacts
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# 7. Expose the port where Uvicorn will listen
EXPOSE 8000

# 8. Launch the FastAPI server binding to all network interfaces (0.0.0.0)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
