#!/bin/bash
# Quick setup script to update API key and restart GraphRAG

echo "🔑 GraphRAG API Key Setup"
echo "=========================="

# Prompt for API key
read -p "Enter your Gemini API Key: " API_KEY

if [ -z "$API_KEY" ]; then
    echo "❌ No API key provided. Exiting."
    exit 1
fi

# Update .env file
echo "📝 Updating .env file..."
cat > .env << EOF
# GraphRAG Environment Configuration
# Google Gemini API Configuration (REQUIRED)
GEMINI_API_KEY=$API_KEY
GOOGLE_API_KEY=$API_KEY

# Server Configuration
PORT=5000

# Other API Keys (Optional - using Gemini only)
OPENAI_API_KEY=demo_mode
GITHUB_MODELS_API_KEY=demo_mode

# Note: API key updated with valid Gemini 2.5 Pro key
EOF

echo "✅ API key updated successfully!"

# Restart the system
echo "🚀 Restarting GraphRAG system with new API key..."
pkill -f "uvicorn app.main:app" 2>/dev/null
pkill -f "python3 -m http.server 8080" 2>/dev/null
sleep 2

echo "🔥 Starting GraphRAG with Gemini 2.5 Pro thinking capabilities..."
./start_graphrag.sh
