#!/usr/bin/env python3
"""Simple test server for Gemini integration"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test basic imports
    print("Testing imports...")
    import os
    print("✅ os imported")
    
    from dotenv import load_dotenv
    print("✅ dotenv imported")
    
    load_dotenv()
    
    # Test Gemini API
    gemini_key = os.getenv("GEMINI_API_KEY")
    print(f"🔑 Gemini API key: {'✅ Found' if gemini_key and gemini_key != 'your_gemini_api_key_here' else '❌ Missing'}")
    
    # Test Google Generative AI
    import google.generativeai as genai
    print("✅ google.generativeai imported")
    
    if gemini_key and gemini_key != 'your_gemini_api_key_here':
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        print("🔥 Gemini 2.5 Flash model initialized successfully!")
        
        # Test a simple query
        response = model.generate_content("Say hello and confirm you're working!")
        print(f"🤖 Gemini response: {response.text}")
    else:
        print("⚠️  Gemini API key not configured properly")
    
    print("\n🎉 All tests passed! Gemini integration is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
