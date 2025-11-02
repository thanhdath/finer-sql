#!/bin/bash

# Install dependencies if needed
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the API server
echo "Starting SQLite API server..."
uvicorn api:app --reload --host 0.0.0.0 --port 8000
