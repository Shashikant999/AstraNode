#!/bin/bash

echo "🔍 Research Paper Knowledge Graph - Status Check"
echo "================================================"
echo ""

# Check backend
echo -n "Backend (port 3001): "
if curl -s http://localhost:3001/api/papers >/dev/null 2>&1; then
    papers=$(curl -s http://localhost:3001/api/papers | jq length 2>/dev/null || echo "?")
    echo "✅ Running ($papers papers loaded)"
else
    echo "❌ Not running"
fi

# Check frontend  
echo -n "Frontend (port 3000): "
if curl -s http://localhost:3000 >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not running"
fi

echo ""
echo "🌐 Application URL: http://localhost:3000"
echo ""

if curl -s http://localhost:3000 >/dev/null 2>&1 && curl -s http://localhost:3001/api/papers >/dev/null 2>&1; then
    echo "🎉 Application is ready to use!"
    echo "   1. Click 'Start AI Analysis' to process your research papers"
    echo "   2. Click 'Ask AI' to use GraphRAG for intelligent querying"
    echo ""
    echo "🤖 GraphRAG Features Available:"
    echo "   • Natural language research queries"
    echo "   • Semantic paper search and discovery"  
    echo "   • AI-powered research insights"
    echo "   • Concept exploration and connections"
else
    echo "⚠️  Some services are not running. Please start the application:"
    echo "   ./start-app.sh"
fi
