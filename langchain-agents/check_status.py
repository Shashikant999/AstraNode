#!/usr/bin/env python3
"""
🔍 Quick Status Check for Google Gemini API
Run this after completing the setup guide
"""

import sys
import os
from dotenv import load_dotenv

def check_status():
    load_dotenv()
    
    print("🔍 Google Gemini API Status Check")
    print("=" * 40)
    
    # Project info
    project_id = "gen-lang-client-0400019191"
    print(f"📋 Project: {project_id}")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ No API key found in .env file")
        return False
        
    print(f"🔑 API Key: {api_key[:10]}...{api_key[-4:]}")
    
    # Test the API
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        print("🧪 Testing API connection...")
        
        # Simple test
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Respond with exactly: 'API Working!'")
        
        result_text = response.text.strip()
        if "API Working" in result_text:
            print("✅ SUCCESS! Gemini API is working perfectly!")
            print(f"🤖 Response: {result_text}")
            
            # Test our research agent
            print("\n🔬 Testing Research Agent...")
            sys.path.append('.')
            from app.agents import LangChainResearchAgent
            
            agent = LangChainResearchAgent()
            if agent.demo_mode:
                print("⚠️  Agent in demo mode (API might have issues)")
            else:
                print("✅ Research Agent connected to live Gemini API!")
                
            # Quick research test
            result = agent.query("What is microgravity?")
            print(f"🧪 Test query successful!")
            print(f"🤖 Provider: {result.get('provider', 'unknown')}")
            
            return True
            
        else:
            print(f"⚠️  Unexpected response: {result_text}")
            return False
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ API Error: {error_msg}")
        
        if "403" in error_msg or "PERMISSION_DENIED" in error_msg:
            print("\n💡 403 Error Solutions:")
            print("   1. Enable Generative Language API:")
            print("      https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
            print(f"   2. Select project: {project_id}")
            
        elif "400" in error_msg or "API_KEY_INVALID" in error_msg:
            print("\n💡 API Key Invalid Solutions:")
            print("   1. Generate new API key:")
            print("      https://aistudio.google.com/app/apikey")
            print(f"   2. Ensure key is for project: {project_id}")
            print("   3. Update .env file with new key")
            
        elif "429" in error_msg or "quota" in error_msg.lower():
            print("\n💡 Quota/Billing Solutions:")
            print("   1. Enable billing:")
            print(f"      https://console.cloud.google.com/billing/linkedaccount?project={project_id}")
            print("   2. Check quotas:")
            print(f"      https://console.cloud.google.com/iam-admin/quotas?project={project_id}")
            
        return False

if __name__ == "__main__":
    print("Run this after completing setup steps in SETUP_GUIDE.md\n")
    
    success = check_status()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 ALL SYSTEMS GO!")
        print("   ➤ Start research system: uv run python -m app.main")
        print("   ➤ Open browser: http://localhost:8000")
    else:
        print("⚠️  Setup needed - check the solutions above")
        print("   ➤ Follow: SETUP_GUIDE.md")
        print("   ➤ Demo mode: uv run python demo_system.py")
