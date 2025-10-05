#!/bin/bash
echo "🚀 Starting NASA Space Apps GraphRAG..."
cd "$(dirname "$0")"
source .venv/bin/activate
cd langchain-agents
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
