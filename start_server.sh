#!/bin/bash
# QIS Dashboard Server Startup Script
# Sets proxy and starts Flask app

echo "Starting QIS Dashboard Server..."

# Set proxy environment variables
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
export NO_PROXY="localhost,127.0.0.1"

echo "[INFO] Proxy configured: $HTTP_PROXY"

# Activate virtual environment
source .venv/Scripts/activate

echo "[INFO] Virtual environment activated"

# Start Flask server
echo "[INFO] Starting Flask server..."
python -B app.py
