#!/bin/bash
# QIS Dashboard Server Stop Script
# Kills Flask app process

echo "Stopping QIS Dashboard Server..."

# Find and kill Python processes running app.py
pkill -f "python.*app.py" && echo "[INFO] Server stopped" || echo "[WARN] No running server found"
