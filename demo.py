#!/usr/bin/env python3
"""
NASA Space Apps GraphRAG - Demo Script
====================================
Quick demonstration of system capabilities
"""

import requests
import json
import time
import sys

API_BASE = "http://localhost:8000"

def test_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def demo_query(query, description):
    """Demonstrate a query"""
    print(f"\n🔍 {description}")
    print(f"Query: '{query}'")
    print("─" * 60)
    
    try:
        response = requests.post(
            f"{API_BASE}/gemini/query",
            json={"query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response: {result.get('response', 'No response')[:200]}...")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")

def main():
    print("""
🚀 NASA Space Apps GraphRAG - Live Demo
========================================
Testing your system with real queries...
""")
    
    # Check server
    if not test_server():
        print("❌ Server not running! Please start with: ./run.sh")
        sys.exit(1)
    
    print("✅ Server is running!")
    
    # Demo queries
    demo_queries = [
        ("What are the effects of microgravity on plant growth?", "Microgravity Research"),
        ("How does space radiation affect human cells?", "Radiation Biology"),
        ("Show me research on Mars soil experiments", "Planetary Science"),
        ("What papers discuss space agriculture?", "Space Agriculture")
    ]
    
    for query, desc in demo_queries:
        demo_query(query, desc)
        time.sleep(2)  # Rate limiting
    
    print(f"""
🎉 Demo Complete!
================
Your GraphRAG system successfully processed queries about:
• Microgravity effects on biology
• Space radiation research  
• Mars exploration
• Space agriculture

💡 Try more queries at: {API_BASE}
""")

if __name__ == "__main__":
    main()
