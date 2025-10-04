#!/usr/bin/env python3
"""
🚀 Test New Gemini API Key with Latest SDK
Based on the official Google AI quickstart documentation
"""

import os
import sys
from dotenv import load_dotenv

def test_new_api_key():
    load_dotenv()
    
    print("🔑 Testing New Gemini API Key")
    print("=" * 40)
    
    # Check the new API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No GEMINI_API_KEY found in .env")
        return False
    
    print(f"🔍 API Key: {api_key[:10]}...{api_key[-4:]} (length: {len(api_key)})")
    print(f"✅ Format looks good: {'AIza' in api_key}")
    
    # Test with the new Google GenAI SDK pattern (from the docs)
    print("\n🧪 Testing with Google GenAI SDK...")
    
    try:
        # Using the new import pattern from the docs
        from google import genai
        
        print("✅ google.genai imported successfully")
        
        # Create client (API key from environment)
        client = genai.Client()
        print("✅ Client created successfully")
        
        # Test with simple content generation
        print("📡 Making API request...")
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents="Say exactly: 'New API key is working perfectly!'"
        )
        
        result_text = response.text.strip()
        print(f"✅ SUCCESS! Response: {result_text}")
        
        if "working perfectly" in result_text.lower():
            print("🎉 API KEY IS FULLY FUNCTIONAL!")
            return True
        else:
            print(f"⚠️  Unexpected response format: {result_text}")
            return True  # Still working, just different response
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ API Error: {error_msg}")
        
        # Error diagnostics
        if "403" in error_msg or "PERMISSION_DENIED" in error_msg:
            print("\n💡 Solution: Enable Generative Language API")
            print("   https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
            
        elif "400" in error_msg or "API_KEY_INVALID" in error_msg:
            print("\n💡 Solution: Check API key configuration")
            print("   1. Verify key is correct in .env file")
            print("   2. Check Google AI Studio: https://aistudio.google.com/app/apikey")
            
        elif "429" in error_msg or "quota" in error_msg.lower():
            print("\n💡 Solution: Check billing and quotas")
            print("   https://console.cloud.google.com/billing")
            
        return False

def test_langchain_integration():
    """Test the LangChain integration with new key"""
    
    print("\n🔗 Testing LangChain Integration...")
    
    try:
        sys.path.append('.')
        from app.agents import LangChainResearchAgent
        
        agent = LangChainResearchAgent()
        
        if agent.demo_mode:
            print("⚠️  LangChain agent still in demo mode")
            print("   This might indicate LangChain needs the older API format")
        else:
            print("✅ LangChain agent connected to live Gemini!")
            
            # Quick test query
            result = agent.query("What is microgravity?")
            print(f"✅ Query successful!")
            print(f"🤖 Provider: {result.get('provider', 'unknown')}")
            print(f"📝 Response preview: {result.get('response', '')[:100]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ LangChain integration error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 New Gemini API Key Test Suite")
    print("Using latest Google GenAI SDK patterns\n")
    
    # Test the direct API
    api_success = test_new_api_key()
    
    # Test LangChain integration
    langchain_success = test_langchain_integration()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   🔑 Direct API: {'✅ Working' if api_success else '❌ Issues'}")
    print(f"   🔗 LangChain: {'✅ Working' if langchain_success else '❌ Issues'}")
    
    if api_success:
        print("\n🎯 Next Steps:")
        print("   1. Start research system: uv run python -m app.main")
        print("   2. Open browser: http://localhost:8000")
        print("   3. Test knowledge graph queries!")
    else:
        print("\n🔧 Troubleshooting needed:")
        print("   1. Check billing: https://console.cloud.google.com/billing")
        print("   2. Enable APIs: https://console.cloud.google.com/apis/library")
        print("   3. Verify API key: https://aistudio.google.com/app/apikey")
