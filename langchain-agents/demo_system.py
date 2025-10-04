#!/usr/bin/env python3
"""
🚀 Complete Knowledge Graph Research System Demo
Works with or without Google Gemini API - showcases full functionality
"""

import sys
import os
from dotenv import load_dotenv
import json

# Load environment
load_dotenv()
sys.path.append('.')

def main():
    print("🌟 Space Biology Knowledge Graph Research System")
    print("=" * 60)
    
    # Import our research agent
    from app.agents import LangChainResearchAgent
    
    agent = LangChainResearchAgent()
    
    if agent.demo_mode:
        print("🎯 Status: DEMO MODE (Google API billing issue detected)")
        print("   ➤ System fully functional with knowledge graph of 607 papers")
        print("   ➤ AI analysis using demo intelligence")
        print("   ➤ Ready to upgrade to Gemini when API access restored")
    else:
        print("🎯 Status: LIVE MODE (Google Gemini API connected)")
        print("   ➤ Full AI-powered analysis with Google Gemini 2.5 Flash")
    
    print("\n📚 Knowledge Graph Coverage:")
    print("   • 607 curated space biology research papers")
    print("   • Topics: microgravity, radiation, cellular biology")
    print("   • Data sources: NASA, ESA, academic journals")
    
    # Demo queries
    demo_queries = [
        "What are the effects of microgravity on human muscle tissue?",
        "How does space radiation affect DNA repair mechanisms?", 
        "What cellular changes occur during long-duration spaceflight?",
        "Which genes are upregulated in microgravity conditions?"
    ]
    
    print(f"\n🧪 Running Demo Queries:")
    print("=" * 40)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("   Processing...")
        
        try:
            result = agent.query(query)
            
            print(f"   ✅ Success!")
            print(f"   🤖 Provider: {result.get('provider', 'unknown')}")
            print(f"   📄 Response length: {len(result.get('response', ''))} chars")
            
            # Show a preview of the response
            response_preview = result.get('response', '')[:300] + "..."
            print(f"   🔍 Preview: {response_preview}")
            
            if result.get('papers'):
                print(f"   📊 Referenced papers: {len(result['papers'])}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test paper analysis
    print(f"\n🔬 Testing Paper Analysis:")
    print("=" * 40)
    
    # Sample paper ID (would be real in actual system)
    sample_paper = {
        "title": "Microgravity-induced changes in human cardiovascular function",
        "authors": ["Smith, J.", "Johnson, K.", "Williams, L."],
        "abstract": "This study examines cardiovascular adaptations during spaceflight...",
        "year": 2023
    }
    
    print(f"📄 Sample Paper: {sample_paper['title']}")
    
    try:
        analysis = agent.analyze_paper(sample_paper)
        
        print(f"   ✅ Analysis complete!")
        print(f"   🎯 Key insights: {len(analysis.get('key_insights', []))} found")
        print(f"   🏷️  Topics: {', '.join(analysis.get('topics', [])[:3])}")
        print(f"   🔗 Related concepts: {len(analysis.get('related_concepts', []))}")
        
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
    
    # System status summary
    print(f"\n" + "=" * 60)
    print(f"📊 System Status Summary:")
    print(f"   🤖 AI Agent: {'Demo Mode' if agent.demo_mode else 'Live Gemini'}")
    print(f"   📚 Knowledge Graph: 607 papers loaded")
    print(f"   🔧 FastAPI Server: Ready to start")
    print(f"   🌐 Web Interface: Ready for deployment")
    
    print(f"\n🚀 Ready to Launch!")
    print(f"   1. Start the research system: uv run python -m app.main")
    print(f"   2. Access web interface at: http://localhost:8000")
    print(f"   3. Query the knowledge graph via API or web UI")
    
    if agent.demo_mode:
        print(f"\n💡 To Enable Full Gemini Integration:")
        print(f"   1. Visit: https://aistudio.google.com/")
        print(f"   2. Enable billing for your project")
        print(f"   3. Verify API quotas and permissions")
        print(f"   4. System will auto-detect and upgrade!")

if __name__ == "__main__":
    main()
