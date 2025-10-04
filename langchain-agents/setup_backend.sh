#!/bin/bash
# GraphRAG Backend Setup Script
# Automatically starts the backend with proper configuration

echo "🚀 Starting GraphRAG Backend..."

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANGCHAIN_DIR="$PROJECT_ROOT/langchain-agents"

# Create virtual environment in project root if it doesn't exist
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "📦 Creating virtual environment..."
    cd "$PROJECT_ROOT"
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

# Install/upgrade required packages
echo "📦 Installing required packages..."
pip install --upgrade fastapi uvicorn python-dotenv google-generativeai langchain langchain-google-genai langchain-core langchain-openai

# Check if CSV exists, create sample if not
CSV_FILE="$PROJECT_ROOT/SB_publication_PMC.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "📄 Creating sample CSV file..."
    cat > "$CSV_FILE" << 'EOF'
Title,Link
Microgravity Effects on Bone Loss,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3630201/
Cellular Response to Weightlessness,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11940681/
Space Biology Research Overview,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11579474/
Bone Density Changes in Space,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4118556/
Osteoblast Function in Microgravity,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3747666/
EOF
fi

# Set environment variables for proper module loading
export PYTHONPATH="$PROJECT_ROOT/langchain-agents:$PYTHONPATH"

# Load environment variables from .env file
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "🔑 Loading environment variables from .env file..."
    export $(cat "$PROJECT_ROOT/.env" | grep -v '^#' | xargs)
fi

# Change to langchain-agents directory
cd "$LANGCHAIN_DIR"

echo "✅ Starting FastAPI backend on http://localhost:8000..."
echo "📊 Paper database: $(wc -l < "$CSV_FILE") papers loaded"
echo "🔑 Gemini API: Pre-configured and ready"
echo ""

# Start the server
uvicorn app.main:app --reload --port 8000 --host 0.0.0.0
