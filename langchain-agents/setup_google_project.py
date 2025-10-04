#!/usr/bin/env python3
"""
🔧 Google Cloud Project Setup for Gemini API
Project: gen-lang-client-0400019191 (899103612067)
"""

import os
import sys
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    print("🚀 Google Cloud Project Setup for Gemini API")
    print("=" * 60)
    
    # Project Information
    project_id = "gen-lang-client-0400019191"
    project_number = "899103612067"
    
    print(f"📋 Project Details:")
    print(f"   Project ID: {project_id}")
    print(f"   Project Number: {project_number}")
    
    # Check current API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    print(f"\n🔑 Current API Key Status:")
    print(f"   GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Missing'}")
    print(f"   GOOGLE_API_KEY: {'✅ Set' if google_key else '❌ Missing'}")
    
    if gemini_key:
        print(f"   Key format: {gemini_key[:10]}...{gemini_key[-4:]} (length: {len(gemini_key)})")
    
    print(f"\n🔍 Diagnosis & Setup Steps:")
    print("=" * 40)
    
    print(f"1. 📊 CHECK BILLING STATUS:")
    print(f"   • Go to: https://console.cloud.google.com/billing")
    print(f"   • Select project: {project_id}")
    print(f"   • Ensure billing is ENABLED")
    print(f"   • Add a valid payment method if needed")
    
    print(f"\n2. 🚀 ENABLE GENERATIVE AI API:")
    print(f"   • Go to: https://console.cloud.google.com/apis/library")
    print(f"   • Search: 'Generative Language API'")
    print(f"   • Click ENABLE for project {project_id}")
    print(f"   • Alternative direct link:")
    print(f"     https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
    
    print(f"\n3. 🔑 VERIFY API KEY SETUP:")
    print(f"   • Go to: https://aistudio.google.com/app/apikey")
    print(f"   • Ensure key is created for project: {project_id}")
    print(f"   • Current key should work with this project")
    
    print(f"\n4. 📋 CHECK QUOTAS & LIMITS:")
    print(f"   • Go to: https://console.cloud.google.com/iam-admin/quotas")
    print(f"   • Filter by: 'Generative Language API'")
    print(f"   • Ensure quotas are sufficient")
    print(f"   • Request increases if needed")
    
    print(f"\n5. 🔒 VERIFY PERMISSIONS:")
    print(f"   • Go to: https://console.cloud.google.com/iam-admin/iam")
    print(f"   • Check your user has 'AI Platform User' role")
    print(f"   • Or 'Editor'/'Owner' permissions")
    
    print(f"\n🧪 Quick Test Commands:")
    print("=" * 40)
    
    # Test with gcloud CLI if available
    print(f"📡 Test API access with gcloud:")
    print(f"   gcloud config set project {project_id}")
    print(f"   gcloud services list --enabled | grep generative")
    print(f"   gcloud auth application-default login")
    
    # Test with curl
    if gemini_key:
        print(f"\n🌐 Test API with curl:")
        curl_cmd = f'''curl -H 'Content-Type: application/json' \\
     -d '{{"contents":[{{"parts":[{{"text":"Hello"}}]}}]}}' \\
     -X POST 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_key[:10]}...{gemini_key[-4:]}'
'''
        print(f"   {curl_cmd}")
    
    print(f"\n✅ Automated Fix Attempt:")
    print("=" * 40)
    
    # Test the current setup
    try:
        print(f"🧪 Testing current configuration...")
        
        import google.generativeai as genai
        
        if gemini_key:
            # Configure with project info
            genai.configure(api_key=gemini_key)
            
            # Try to list models to test authentication
            try:
                models = list(genai.list_models())
                print(f"   ✅ API Authentication successful!")
                print(f"   📊 Available models: {len(models)}")
                
                # Test a simple generation
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content("Say 'Project setup successful!'")
                print(f"   🎉 Generation test: {response.text}")
                
                return True
                
            except Exception as e:
                error_msg = str(e)
                print(f"   ❌ API call failed: {error_msg}")
                
                if "403" in error_msg:
                    print(f"   💡 403 Error = Enable the Generative Language API")
                    print(f"      Link: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project={project_id}")
                    
                elif "401" in error_msg or "API_KEY_INVALID" in error_msg:
                    print(f"   💡 Invalid API Key = Check key is for project {project_id}")
                    print(f"      Link: https://aistudio.google.com/app/apikey")
                    
                elif "429" in error_msg or "quota" in error_msg.lower():
                    print(f"   💡 Quota exceeded = Check billing and quotas")
                    print(f"      Billing: https://console.cloud.google.com/billing")
                    print(f"      Quotas: https://console.cloud.google.com/iam-admin/quotas")
                    
                return False
        else:
            print(f"   ❌ No API key found in environment")
            return False
            
    except ImportError:
        print(f"   ❌ google-generativeai not installed")
        print(f"   💡 Run: uv add google-generativeai")
        return False
    except Exception as e:
        print(f"   ❌ Setup error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    print(f"\n" + "=" * 60)
    if success:
        print(f"🎉 SUCCESS! Your Gemini API is working!")
        print(f"   ➤ Run the research system: uv run python -m app.main")
        print(f"   ➤ Test integration: uv run python test_enhanced.py")
    else:
        print(f"⚠️  Setup needed. Follow the steps above.")
        print(f"   ➤ Most common fix: Enable billing + Enable Generative Language API")
        print(f"   ➤ System works in demo mode: uv run python demo_system.py")
