#!/bin/bash

echo "🧪 Testing GraphRAG functionality..."
echo ""

# Step 1: Trigger analysis
echo "📊 Step 1: Analyzing papers..."
ANALYSIS_RESULT=$(curl -s -X POST http://localhost:3001/api/analyze)

if echo "$ANALYSIS_RESULT" | grep -q '"success":true'; then
    echo "✅ Analysis completed successfully"
    
    # Extract some stats
    PAPERS=$(echo "$ANALYSIS_RESULT" | grep -o '"papersCount":[0-9]*' | cut -d: -f2)
    CONNECTIONS=$(echo "$ANALYSIS_RESULT" | grep -o '"connectionsCount":[0-9]*' | cut -d: -f2)
    
    echo "   Papers analyzed: $PAPERS"
    echo "   Connections found: $CONNECTIONS"
else
    echo "❌ Analysis failed"
    echo "Response: $ANALYSIS_RESULT"
    exit 1
fi

echo ""

# Step 2: Test GraphRAG query
echo "🤖 Step 2: Testing GraphRAG query..."
RAG_QUERY='{"query": "microgravity effects on bone density", "options": {"maxResults": 3, "useGraphStructure": true}}'

RAG_RESULT=$(curl -s -X POST http://localhost:3001/api/rag/query \
  -H "Content-Type: application/json" \
  -d "$RAG_QUERY")

if echo "$RAG_RESULT" | grep -q '"query"'; then
    echo "✅ GraphRAG query successful"
    
    # Extract some results
    RESULT_COUNT=$(echo "$RAG_RESULT" | grep -o '"results":\[' | wc -l)
    echo "   Results returned: $RESULT_COUNT"
    
    # Show first result title if available
    FIRST_TITLE=$(echo "$RAG_RESULT" | grep -o '"title":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [ ! -z "$FIRST_TITLE" ]; then
        echo "   First result: $FIRST_TITLE"
    fi
else
    echo "❌ GraphRAG query failed"
    echo "Response: $RAG_RESULT"
fi

echo ""

# Step 3: Test concept exploration
echo "🔍 Step 3: Testing concept exploration..."
CONCEPT_RESULT=$(curl -s "http://localhost:3001/api/rag/concept/microgravity")

if echo "$CONCEPT_RESULT" | grep -q '"papers"'; then
    echo "✅ Concept exploration successful"
    CONCEPT_PAPERS=$(echo "$CONCEPT_RESULT" | grep -o '"papers":\[' | wc -l)
    echo "   Related papers found: $CONCEPT_PAPERS"
else
    echo "❌ Concept exploration failed"
    echo "Response: $CONCEPT_RESULT"
fi

echo ""
echo "🎉 GraphRAG testing complete!"
echo ""
echo "🌐 Try the web interface:"
echo "   1. Open http://localhost:3000"
echo "   2. Click 'Ask AI' button"  
echo "   3. Ask: 'What research exists on microgravity and bone health?'"
