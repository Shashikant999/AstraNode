#!/usr/bin/env python3
"""Test script for LangChain Research Agents"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from app.main import app

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting LangChain Research Agents server...")
    print("📊 GraphRAG integration: Connecting to http://localhost:3001")
    print("🤖 Available agents: research_assistant, concept_explorer, collaboration_finder, analysis_specialist")
    print("🔗 API docs will be at: http://localhost:8000/docs")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
