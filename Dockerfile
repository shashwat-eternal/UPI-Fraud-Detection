# UPI Fraud Detection API — container image
# Build:  docker build -t upi-fraud-api .
# Run:    docker run -p 8000:8000 upi-fraud-api

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY src/ src/
COPY models/ models/
COPY frontend/ frontend/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
