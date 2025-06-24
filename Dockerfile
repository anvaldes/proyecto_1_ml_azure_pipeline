FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY train.py .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
