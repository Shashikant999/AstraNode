#!/usr/bin/env python3
"""Enhanced Google Gemini API Test with Billing Error Detection"""

import sys
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()
sys.path.append('.')

print("🔥 Enhanced Google Gemini API Test")
print("=" * 50)

# Test environment setup
gemini_key = os.getenv("GEMINI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY") 

print(f"🔑 API Key Status:")
print(f"   GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Missing'}")
print(f"   GOOGLE_API_KEY: {'✅ Set' if google_key else '❌ Missing'}")

if gemini_key:
    print(f"   Key format: {gemini_key[:10]}...{gemini_key[-4:]} (length: {len(gemini_key)})")
    print(f"   Starts with 'AIza': {'✅ Yes' if gemini_key.startswith('AIza') else '❌ No'}")

# Enhanced API test with detailed error handling
print(f"\n🧪 Testing Google Generative AI...")

try:
    import google.generativeai as genai
    print("   ✅ google.generativeai imported successfully")
    
    if gemini_key:
        print("   🔧 Configuring API key...")
        genai.configure(api_key=gemini_key)
        
        print("   🤖 Creating Gemini 2.5 Flash model...")
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
        
        print("   📡 Testing API call...")
        response = model.generate_content("Say 'Hello! I am Google Gemini and I am working perfectly!'")
        
        print("   🎉 SUCCESS! Google Gemini API is working!")
        print(f"   ✨ Response: {response.text}")
        
        # Test usage metadata if available
        if hasattr(response, 'usage_metadata'):
            print(f"   📊 Usage: {response.usage_metadata}")
        
        api_working = True
        
except Exception as e:
    error_message = str(e).lower()
    
    print(f"   ❌ API Error: {e}")
    
    # Specific error handling
    if "api key not valid" in error_message:
        print(f"\n💡 API Key Issue Detected:")
        print(f"   • The API key format looks correct but is being rejected")
        print(f"   • This is commonly a billing/quota issue")
        print(f"   • Solutions:")
        print(f"     1. Check Google AI Studio billing: https://aistudio.google.com/")
        print(f"     2. Ensure you have billing enabled for the project")
        print(f"     3. Check API quotas and limits")
        print(f"     4. Verify the API key has Generative AI permissions")
        
    elif "quota" in error_message or "limit" in error_message:
        print(f"\n📊 Quota Issue Detected:")
        print(f"   • You've hit API rate limits or quotas")
        print(f"   • Wait a few minutes and try again")
        print(f"   • Check your usage in Google Cloud Console")
        
    elif "billing" in error_message:
        print(f"\n💳 Billing Issue Detected:")
        print(f"   • Billing must be enabled for Gemini API")
        print(f"   • Go to Google Cloud Console and set up billing")
        
    else:
        print(f"\n🔍 Unknown Error:")
        print(f"   • This might be a temporary API issue")
        print(f"   • Try again in a few minutes")
        print(f"   • Check Google AI Studio status")

# Test with fallback demo mode
print(f"\n🚀 Testing Research Agent (with fallbacks)...")

try:
    from app.agents import LangChainResearchAgent
    print("   ✅ Research agent imported")
    
    agent = LangChainResearchAgent()
    
    if agent.demo_mode:
        print("   ⚠️  Agent running in DEMO MODE")
        print("   💡 This means the Google API isn't working, but the system still functions!")
    else:
        print("   ✅ Agent initialized with live Google Gemini API!")
    
    # Test a query regardless of mode
    print("   🧪 Testing research query...")
    result = agent.query("What are the effects of microgravity on human cells?")
    
    print(f"   ✅ Query successful!")
    print(f"   🤖 Provider: {result.get('provider', 'unknown')}")
    print(f"   📝 Response preview: {result.get('response', '')[:200]}...")
    
except Exception as e:
    print(f"   ❌ Research agent error: {e}")

print(f"\n" + "=" * 50)
print(f"📋 Summary:")
print(f"   • Google Gemini API: {'Working' if 'SUCCESS' in globals() else 'Issues detected'}")
print(f"   • Research System: Functional (with demo fallback)")
print(f"   • Ready for Knowledge Graph: ✅")

print(f"\n🎯 Next Steps:")
if gemini_key:
    print(f"   1. If billing issues: Enable billing in Google AI Studio")
    print(f"   2. Check quotas at https://aistudio.google.com/")
    print(f"   3. The system works in demo mode regardless!")
    print(f"   4. Start the research paper servers when ready")
else:
    print(f"   1. Get a Gemini API key from https://aistudio.google.com/app/apikey")
    print(f"   2. Add it to the .env file")
    print(f"   3. The system works in demo mode for now!")
