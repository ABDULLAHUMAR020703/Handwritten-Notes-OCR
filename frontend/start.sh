#!/bin/bash
# Start script for frontend

echo "Starting Handwritten Notes OCR Frontend..."

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start the development server
echo "Starting development server on http://localhost:3000"
npm run dev
