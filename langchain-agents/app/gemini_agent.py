"""Google Gemini Integration for Research Paper Analysis"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

class GeminiResearchAgent:
    """Research agent using Google Gemini API directly"""
    
    def __init__(self, api_key: str = None):
        """Initialize with Gemini API"""
        # Debug environment variables
        gemini_key = os.getenv("GEMINI_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        
        print(f"🔧 Debug - GEMINI_API_KEY present: {'Yes' if gemini_key else 'No'}")
        print(f"🔧 Debug - GOOGLE_API_KEY present: {'Yes' if google_key else 'No'}")
        print(f"🔧 Debug - Provided API key: {'Yes' if api_key else 'No'}")
        
        # Use provided API key, environment variable, or hardcoded key
        self.api_key = api_key or gemini_key or google_key or "AIzaSyCaonefFVlB5g3xpg9vipm0FTZJSDtcQEM"
        
        if not self.api_key or self.api_key == "your_gemini_api_key_here" or self.api_key == "demo_mode":
            print("⚠️  No valid API key found - running in DEMO MODE")
            self.api_working = False
            return
        
        # Import here to handle optional dependency
        try:
            print("🔧 Debug - Importing google.generativeai...")
            import google.generativeai as genai
            print("🔧 Debug - google.generativeai imported successfully")
            
            print(f"🔧 Debug - Configuring with API key: {self.api_key[:10]}...")
            genai.configure(api_key=self.api_key)
            print("🔧 Debug - API key configured successfully")
            
            # Initialize Gemini Pro model
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            
            # Test the API key with a simple call
            try:
                test_response = self.model.generate_content("Hello")
                print(f"✅ Google Gemini Pro initialized and tested successfully!")
                self.api_working = True
            except Exception as e:
                print(f"⚠️  Gemini API key validation failed: {e}")
                print(f"🔧 API Key present: {'Yes' if self.api_key else 'No'}")
                print(f"🔧 API Key length: {len(self.api_key) if self.api_key else 0}")
                print("💡 Please get a valid API key from https://aistudio.google.com/app/apikey")
                self.api_working = False
            
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        except Exception as e:
            print(f"⚠️  Failed to initialize Gemini: {e}")
            self.api_working = False
    
    def analyze_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a research paper using Gemini"""
        
        paper_text = f"""
        Title: {paper_data.get('title', 'Unknown')}
        Authors: {paper_data.get('authors', 'Unknown')}
        Abstract: {paper_data.get('abstract', 'No abstract available')}
        Keywords: {paper_data.get('keywords', 'No keywords')}
        """
        
        prompt = f"""
        As an expert in space biology and microgravity research, analyze this research paper and provide:
        
        1. **Key Concepts**: Extract 5-8 main scientific concepts
        2. **Research Domain**: Classify the primary research area
        3. **Methodology**: Identify research methods used
        4. **Significance**: Rate importance (1-10) and explain
        5. **Connections**: Suggest related research areas
        6. **Future Work**: Identify potential research directions
        
        Paper to analyze:
        {paper_text}
        
        Please provide a structured JSON response.
        """
        
        if not self.api_working:
            return {
                'success': False,
                'error': 'Gemini API not properly configured. Please set a valid GEMINI_API_KEY.',
                'demo_analysis': self._get_demo_analysis(paper_data),
                'model': 'gemini-2.5-flash',
                'provider': 'google_demo'
            }
        
        try:
            response = self.model.generate_content(prompt)
            return {
                'success': True,
                'analysis': response.text,
                'model': 'gemini-2.5-flash',
                'provider': 'google'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'demo_analysis': self._get_demo_analysis(paper_data),
                'model': 'gemini-2.5-flash',
                'provider': 'google_fallback'
            }
    
    def query_knowledge_graph(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Query the research knowledge graph using Gemini with real paper database"""
        
        # Import and search the actual paper database
        try:
            from .paper_database import get_paper_database, search_research_papers, get_topic_analysis
            
            # Get relevant papers from the database
            db = get_paper_database()
            relevant_papers = db.search_papers(query, max_results=15)
            
            # Get topic analysis
            topic_analysis = db.get_papers_by_topic(query)
            
            # Prepare context from real papers
            paper_context = "\n".join([
                f"- {paper.title} (PMC: {paper.pmc_id})"
                for paper in relevant_papers[:10]
            ])
            
            context_info = f"""
Research Database Context:
- Total Papers: 607 space biology research papers
- Relevant Papers Found: {len(relevant_papers)}
- Categories: {', '.join([k for k, v in topic_analysis['categories'].items() if v])}

Top Relevant Papers:
{paper_context}
"""
        except ImportError:
            context_info = "Context: 607 papers loaded from space biology database"
        
        prompt = f"""
You are an expert research assistant with access to a curated database of 607 space biology papers from PMC (PubMed Central).

{context_info}

User Query: {query}

Please provide a comprehensive analysis based ONLY on the space biology research database:

1. **Direct Answer**: Answer the query using insights from the relevant papers listed above
2. **Paper Connections**: Explain how the found papers relate to each other and address the query
3. **Research Insights**: Key findings and patterns from the paper titles and known research areas
4. **Follow-up Directions**: Suggest specific research questions based on gaps in the current database
5. **Connected Research**: Identify related concepts and methodologies from the paper database

Important: Base your response on the actual paper titles and research areas from our 607-paper space biology database. Mention specific paper titles when relevant.

Focus on: microgravity effects, space biology, life sciences in space, radiation biology, bone/muscle research, cellular responses, gene expression, and related space medicine topics.
        """
        
        if not self.api_working:
            return {
                'success': True,
                'response': self._get_demo_response(query, "query"),
                'model': 'gemini-2.5-flash',
                'query': query,
                'provider': 'google_demo'
            }
        
        try:
            response = self.model.generate_content(prompt)
            return {
                'success': True,
                'response': response.text,
                'model': 'gemini-2.5-flash',
                'query': query,
                'provider': 'google'
            }
        except Exception as e:
            return {
                'success': True,
                'response': self._get_demo_response(query, "query"),
                'error': str(e),
                'model': 'gemini-2.5-flash',
                'query': query,
                'provider': 'google_fallback'
            }
    
    def explore_concept(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """Explore a scientific concept in depth"""
        
        prompt = f"""
        As a space biology research expert, provide a comprehensive exploration of: {concept}
        
        Please cover:
        1. **Definition & Context**: Clear explanation in space biology
        2. **Current Research**: Key findings and methodologies
        3. **Microgravity Effects**: How this concept relates to space environments
        4. **Research Gaps**: Areas needing more investigation
        5. **Interdisciplinary Connections**: Related fields and concepts
        6. **Future Directions**: Emerging research opportunities
        
        Depth level: {depth} (1=basic, 2=intermediate, 3=advanced)
        """
        
        try:
            response = self.model.generate_content(prompt)
            return {
                'success': True,
                'exploration': response.text,
                'concept': concept,
                'depth': depth,
                'model': 'gemini-2.5-flash',
                'provider': 'google'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'concept': concept,
                'depth': depth,
                'model': 'gemini-2.5-flash',
                'provider': 'google'
            }
    
    def find_collaborations(self, research_interest: str) -> Dict[str, Any]:
        """Find potential collaboration opportunities"""
        
        prompt = f"""
        Based on the research interest: "{research_interest}"
        
        Analyze potential collaboration opportunities in space biology research:
        
        1. **Research Groups**: Identify relevant research institutions and labs
        2. **Key Researchers**: Notable scientists in this area
        3. **Funding Opportunities**: Relevant grants and programs (NASA, ESA, etc.)
        4. **Conference Networks**: Key conferences and symposiums
        5. **Interdisciplinary Connections**: Related fields for collaboration
        6. **Current Projects**: Ongoing research initiatives to join
        
        Focus on space biology, microgravity research, and life sciences.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return {
                'success': True,
                'collaborations': response.text,
                'research_interest': research_interest,
                'model': 'gemini-2.5-flash',
                'provider': 'google'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'research_interest': research_interest,
                'model': 'gemini-2.5-flash',
                'provider': 'google'
            }


    def _get_demo_analysis(self, paper_data: Dict[str, Any]) -> str:
        """Provide demo analysis when API is not available"""
        title = paper_data.get('title', 'Unknown Paper')
        abstract = paper_data.get('abstract', 'No abstract available')
        
        return f"""
        **Demo Analysis for: {title}**
        
        🔬 **Key Concepts**: Based on the title and content, this paper likely involves:
        - Space biology research
        - Microgravity effects on biological systems
        - Life sciences in space environments
        
        📊 **Research Domain**: Space Biology & Life Sciences
        
        🧪 **Methodology**: Experimental research methods typical for space biology
        
        ⭐ **Significance**: High importance (8/10) - Contributes to understanding life in space
        
        🔗 **Connections**: Related to other microgravity research and space medicine
        
        🚀 **Future Work**: Potential for follow-up experiments and applications
        
        ℹ️  *This is a demo analysis. For detailed AI-powered analysis, please configure a valid Gemini API key from https://aistudio.google.com/app/apikey*
        """
    
    def _get_demo_response(self, query: str, response_type: str = "query") -> str:
        """Provide demo responses when API is not available"""
        demos = {
            "query": f"""
            **Demo Response for: "{query}"**
            
            Based on our knowledge graph of 607 space biology papers, here's what I can tell you:
            
            🔬 **Research Insights**: This topic is well-represented in space biology literature, with connections to microgravity effects, cellular biology, and space medicine.
            
            📊 **Key Findings**: Multiple studies have explored this area, showing significant impacts of space environments on biological systems.
            
            🧪 **Methodological Approaches**: Researchers typically use ground-based simulators, parabolic flights, and ISS experiments.
            
            🔗 **Related Research**: Connected to broader themes in space biology, astrobiology, and space medicine.
            
            ℹ️  *This is a demo response. For detailed AI analysis with the latest Gemini 2.5 Flash model, please configure a valid API key.*
            """,
            "collaboration": f"""
            **Collaboration Opportunities for: "{query}"**
            
            🏢 **Research Institutions**: NASA Ames Research Center, ESA, JAXA, and major universities with space biology programs
            
            👥 **Key Researchers**: Leading scientists in space biology and microgravity research
            
            💰 **Funding**: NASA Space Biology, ESA Life Sciences, and NSF opportunities
            
            🎯 **Conferences**: COSPAR, IAC, ASGSR, and space biology symposiums
            
            🔬 **Interdisciplinary**: Connections with medicine, engineering, and astrobiology
            
            ℹ️  *For personalized collaboration matching, please configure Gemini API.*
            """,
            "concept": f"""
            **Concept Exploration: "{query}"**
            
            📚 **Definition**: This concept is central to space biology research and understanding life beyond Earth
            
            🔬 **Current Research**: Active area with ongoing experiments on ISS and ground facilities
            
            🌌 **Space Applications**: Critical for long-duration missions and space settlement
            
            🧬 **Biological Impact**: Affects cellular, molecular, and physiological processes
            
            🚀 **Future Directions**: Key area for Mars missions and deep space exploration
            
            ℹ️  *For advanced concept analysis, please set up Gemini API integration.*
            """
        }
        return demos.get(response_type, demos["query"])


def create_gemini_agent(api_key: str = None) -> GeminiResearchAgent:
    """Factory function to create a Gemini research agent"""
    return GeminiResearchAgent(api_key=api_key)
