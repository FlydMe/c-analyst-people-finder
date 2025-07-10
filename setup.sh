#!/bin/bash

echo "🚀 Setting up VC Analyst Search Tool API"
echo "========================================"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys:"
    echo "   - OPENAI_API_KEY (required)"
    echo "   - GOOGLE_API_KEY (required)"
    echo "   - GOOGLE_SEARCH_ENGINE_ID (required)"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: uvicorn main:app --reload"
echo "3. Visit: http://localhost:8000/docs"
