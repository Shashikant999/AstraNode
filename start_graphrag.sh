#!/bin/bash
# GraphRAG Complete System Startup Script
# One command to start everything

echo "🧬 GraphRAG - Intelligent Research Graph System"
echo "=============================================="

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make scripts executable
chmod +x "$PROJECT_ROOT/langchain-agents/setup_backend.sh"

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down GraphRAG system..."
    pkill -f "uvicorn app.main:app"
    pkill -f "python3 -m http.server 8080"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend in background
echo "🚀 Starting backend server..."
cd "$PROJECT_ROOT/langchain-agents"
./setup_backend.sh &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 10

# Start frontend server in background
echo "🌐 Starting frontend server..."
cd "$PROJECT_ROOT/langchain-agents"
python3 -m http.server 8080 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 3

echo ""
echo "✅ GraphRAG System is now running!"
echo ""
echo "🌐 Frontend: http://localhost:8080/graphrag_test.html"
echo "🔗 Backend API: http://localhost:8000"
echo "📊 Health Check: http://localhost:8000/health"
echo ""
echo "🎯 Click 'Run Full Test' to test the complete system"
echo "📱 Press Ctrl+C to stop all services"
echo ""

# Open browser if available
if command -v open >/dev/null 2>&1; then
    echo "🌐 Opening browser..."
    open "http://localhost:8080/graphrag_test.html"
elif command -v xdg-open >/dev/null 2>&1; then
    echo "🌐 Opening browser..."
    xdg-open "http://localhost:8080/graphrag_test.html"
fi

# Keep script running
wait
