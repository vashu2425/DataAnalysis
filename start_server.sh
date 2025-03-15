#!/bin/bash

# Kill any process using port 8000
echo "Stopping any existing server on port 8000..."
sudo kill -9 $(sudo lsof -t -i:8000) 2>/dev/null || true

# Wait a moment to ensure the port is released
sleep 2

# Start the server
echo "Starting server..."
python -m uvicorn backend.app.main:app --reload --port 8000 