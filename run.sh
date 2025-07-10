#!/bin/bash

# Start the FastAPI server
echo "🚀 Starting VC Analyst Search Tool API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | xargs)
fi

# Start the server
echo "🌟 Starting server on http://localhost:8000"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
