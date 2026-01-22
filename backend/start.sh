#!/bin/bash
# Start script for backend

echo "Starting Handwritten Notes OCR Backend..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Create necessary directories
mkdir -p uploads outputs

# Start the server
echo "Starting server on http://localhost:8000"
python -m app.main
