#!/usr/bin/env python3
"""Test Google Gemini API integration with LangChain"""

import sys
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()
sys.path.append('.')

print("🚀 Testing Google Gemini + LangChain Integration")
print("=" * 50)

# Test 1: Environment Variables
print("\n1. Testing Environment Variables:")
gemini_key = os.getenv("GEMINI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY") 
print(f"   GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Not set'}")
print(f"   GOOGLE_API_KEY: {'✅ Set' if google_key else '❌ Not set'}")

if gemini_key:
    print(f"   Key format: {gemini_key[:10]}...{gemini_key[-4:]} (length: {len(gemini_key)})")

# Test 2: Direct Google Generative AI
print("\n2. Testing Google GenerativeAI (Direct):")
try:
    import google.generativeai as genai
    print("   ✅ google.generativeai imported")
    
    if gemini_key:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content("Hello! Can you confirm you're working?")
        print(f"   ✅ Direct API test successful!")
        print(f"   Response: {response.text[:100]}...")
    else:
        print("   ❌ No API key to test")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: LangChain Google Generative AI
print("\n3. Testing LangChain Google GenerativeAI:")
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    print("   ✅ LangChain imports successful")
    
    if google_key:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_key,
            temperature=0.1
        )
        
        messages = [HumanMessage(content="Hello from LangChain! Are you working?")]
        response = llm.invoke(messages)
        print("   ✅ LangChain ChatGoogleGenerativeAI test successful!")
        print(f"   Response: {response.content[:100]}...")
        print(f"   Usage: {getattr(response, 'usage_metadata', 'No usage data')}")
    else:
        print("   ❌ No Google API key to test")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Research Agent
print("\n4. Testing Research Agent:")
try:
    from app.agents_new import LangChainResearchAgent
    print("   ✅ Research agent imported")
    
    agent = LangChainResearchAgent()
    if not agent.demo_mode:
        print("   ✅ Agent initialized successfully!")
        
        # Test query
        result = agent.query("What are the effects of microgravity on cellular biology?")
        print(f"   ✅ Query test successful!")
        print(f"   Response preview: {result.get('response', '')[:150]}...")
    else:
        print("   ⚠️  Agent in demo mode (API key issue)")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 5: FastAPI Server
print("\n5. Testing FastAPI Integration:")
try:
    from app.main import app
    print("   ✅ FastAPI app imported successfully")
    
    # Check health endpoint data
    import asyncio
    async def test_health():
        from app.main import root
        health_data = await root()
        return health_data
    
    health = asyncio.run(test_health())
    print(f"   ✅ Health check: {health.get('status', 'unknown')}")
    print(f"   LangChain available: {health.get('langchain_available', False)}")
    print(f"   Gemini available: {health.get('gemini_available', False)}")
    print(f"   Google API configured: {health.get('api_providers', {}).get('google_api_configured', False)}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 50)
print("🎯 Integration Test Complete!")

if gemini_key and gemini_key.startswith("AIza"):
    print("✅ Google Gemini API properly configured")
    print("🚀 Ready to start the research paper knowledge graph system!")
else:
    print("⚠️  Please check your Google API key configuration")
