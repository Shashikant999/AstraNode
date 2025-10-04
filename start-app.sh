#!/bin/bash

echo "🚀 Starting Research Paper Knowledge Graph Application..."
echo ""

# Check if ports are available
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 3000 is already in use. Stopping existing process..."
    pkill -f "react-scripts start" || true
    sleep 2
fi

if lsof -Pi :3001 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 3001 is already in use. Stopping existing process..."
    pkill -f "node server.js" || true
    sleep 2
fi

echo "📊 Starting backend server (port 3001)..."
node server.js &
BACKEND_PID=$!

echo "⏳ Waiting for backend to start..."
sleep 3

echo "🎨 Starting React frontend (port 3000)..."
cd client && BROWSER=none npm start &
FRONTEND_PID=$!

echo ""
echo "✅ Application started successfully!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend:  http://localhost:3001"
echo ""
echo "📝 To stop the application, press Ctrl+C"
echo ""

# Wait for user interrupt
trap 'echo ""; echo "🛑 Stopping application..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' INT

wait
