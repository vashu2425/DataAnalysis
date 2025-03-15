#!/bin/bash

# Kill all running Uvicorn processes
echo "Killing all running Uvicorn processes..."
pkill -9 -f "uvicorn"

# Wait a moment to ensure all processes are terminated
sleep 2

# Try different ports until we find one that works
for port in 8000 8001 8002 8003 8004 8005; do
    echo "Attempting to start server on port $port..."
    python -m uvicorn app.main:app --reload --port $port &
    
    # Wait a moment to see if the server starts successfully
    sleep 3
    
    # Check if the server is running on this port
    if curl -s http://localhost:$port/docs > /dev/null; then
        echo "Server successfully started on port $port"
        echo "Access the API documentation at http://localhost:$port/docs"
        break
    else
        echo "Failed to start on port $port, trying next port..."
        pkill -9 -f "uvicorn app.main:app --reload --port $port"
        sleep 2
    fi
done 