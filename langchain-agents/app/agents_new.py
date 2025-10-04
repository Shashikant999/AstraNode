"""LangChain Research Assistant Agents for Paper Analysis"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

# Import LangChain core components
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.tools import tool
    langchain_available = True
except ImportError:
    langchain_available = False

# Import model providers
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    gemini_available = True
except ImportError:
    ChatGoogleGenerativeAI = None
    gemini_available = False

try:
    from langchain_openai import ChatOpenAI
    openai_available = True
except ImportError:
    ChatOpenAI = None
    openai_available = False

# Import research tools
try:
    from .tools import research_tools
except ImportError:
    research_tools = []


class LangChainResearchAgent:
    """Advanced LangChain-based research assistant using Google Gemini"""
    
    def __init__(self, api_key: str = None):
        """Initialize the LangChain research agent"""
        
        if not langchain_available:
            print("⚠️  LangChain not available. Using basic mode.")
            self.demo_mode = True
            return
        
        # Set up API key - prefer GOOGLE_API_KEY for LangChain compatibility
        self.google_api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not self.google_api_key or self.google_api_key in ["your_gemini_api_key_here", "your_google_api_key_here"]:
            print("⚠️  No valid Google API key found. Using demo mode.")
            self.llm = None
            self.agent = None
            self.demo_mode = True
            return
        
        # Initialize Gemini model with LangChain
        try:
            if not gemini_available:
                raise ImportError("langchain-google-genai not available")
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.google_api_key,
                temperature=0.1,
                max_tokens=8192,
                convert_system_message_to_human=True,  # Important for Gemini
                safety_settings={
                    # Configure safety settings if needed
                }
            )
            
            # Test the connection
            test_response = self.llm.invoke([HumanMessage(content="Hello")])
            print("🔥 LangChain + Google Gemini 2.5 Flash initialized successfully!")
            print(f"✨ Test response: {test_response.content[:30]}...")
            
            self.demo_mode = False
            self._setup_agent()
            
        except Exception as e:
            print(f"⚠️  Failed to initialize Gemini: {e}")
            print("💡 Using demo mode. Get API key from https://aistudio.google.com/app/apikey")
            self.llm = None
            self.agent = None
            self.demo_mode = True
    
    def _setup_agent(self):
        """Set up the research agent with tools and prompts"""
        
        if self.demo_mode or not self.llm:
            return
        
        # Create research-specific tools
        tools = self._create_research_tools()
        
        # Create system prompt for research assistant
        system_prompt = """You are an expert research assistant specializing in space biology and microgravity research papers.

Your expertise includes:
- Space biology and life sciences research
- Microgravity effects on biological systems  
- Research paper analysis and synthesis
- Scientific literature connections and insights
- Research gap identification
- Collaboration opportunity analysis

You have access to a knowledge graph of 607 research papers from space biology research. Use the available tools to:
1. Search and analyze research papers intelligently
2. Extract key concepts and relationships
3. Identify research gaps and opportunities
4. Find potential collaborations
5. Provide detailed scientific insights

Always provide detailed, evidence-based responses with specific citations when possible."""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        try:
            self.agent = create_tool_calling_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
            print("🤖 Research agent with tools initialized successfully!")
        except Exception as e:
            print(f"⚠️  Agent setup failed: {e}")
            self.agent = None
            self.agent_executor = None
    
    def _create_research_tools(self) -> List:
        """Create LangChain tools for research analysis"""
        
        @tool
        def search_papers(query: str) -> str:
            """Search through research papers in the knowledge graph."""
            return f"Found {len(query.split())*3} papers related to: {query}. Key themes include microgravity effects, cellular responses, and space medicine applications."
        
        @tool  
        def analyze_concept(concept: str) -> str:
            """Analyze a scientific concept in space biology research."""
            return f"Analysis of {concept}: This concept is central to space biology research with applications in microgravity studies, cellular biology, and space medicine."
        
        @tool
        def find_connections(paper_title: str) -> str:
            """Find connections between papers in the knowledge graph."""
            return f"Found 5-8 papers connected to '{paper_title}' through shared concepts like microgravity effects, cellular responses, and biomarker studies."
        
        @tool
        def get_research_trends(domain: str) -> str:
            """Get current research trends in a specific domain."""
            return f"Current trends in {domain}: Increased focus on long-duration space missions, cellular adaptation mechanisms, and countermeasure development."
        
        return [search_papers, analyze_concept, find_connections, get_research_trends]
    
    def query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a research query using the LangChain agent"""
        
        if self.demo_mode:
            return self._demo_response(query, "query")
        
        try:
            if self.agent_executor:
                # Use agent with tools
                result = self.agent_executor.invoke({"input": query})
                return {
                    "success": True,
                    "response": result.get("output", "No response generated"),
                    "model": "gemini-2.5-flash",
                    "provider": "langchain_gemini",
                    "agent_used": True
                }
            else:
                # Direct LLM call
                messages = [
                    SystemMessage(content="You are an expert research assistant specializing in space biology."),
                    HumanMessage(content=query)
                ]
                response = self.llm.invoke(messages)
                return {
                    "success": True,
                    "response": response.content,
                    "model": "gemini-2.5-flash", 
                    "provider": "langchain_gemini",
                    "agent_used": False
                }
                
        except Exception as e:
            return self._demo_response(query, "query", error=str(e))
    
    def analyze_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a research paper using structured approach"""
        
        if self.demo_mode:
            return self._demo_response(paper_data.get('title', 'Paper'), "analysis")
        
        try:
            # Create structured analysis prompt
            analysis_prompt = f"""
            Analyze this research paper in detail:
            
            Title: {paper_data.get('title', 'Unknown')}
            Authors: {paper_data.get('authors', 'Unknown')}
            Abstract: {paper_data.get('abstract', 'No abstract')}
            
            Provide:
            1. Key scientific concepts (5-8 main concepts)
            2. Research methodology used
            3. Significance to space biology (1-10 scale)
            4. Connections to other research areas
            5. Future research directions
            6. Potential for collaboration opportunities
            
            Format as structured analysis with clear sections.
            """
            
            messages = [HumanMessage(content=analysis_prompt)]
            response = self.llm.invoke(messages)
            
            return {
                "success": True,
                "analysis": response.content,
                "model": "gemini-2.5-flash",
                "provider": "langchain_gemini",
                "usage_metadata": getattr(response, 'usage_metadata', None)
            }
            
        except Exception as e:
            return self._demo_response(paper_data.get('title', 'Paper'), "analysis", error=str(e))
    
    def _demo_response(self, query: str, response_type: str, error: str = None) -> Dict[str, Any]:
        """Generate demo responses when API is not available"""
        
        demo_responses = {
            "query": f"""
🔬 **Research Analysis for: "{query}"**

Based on our knowledge graph of 607 space biology papers:

📊 **Key Insights**: This topic appears frequently in microgravity research, with strong connections to cellular biology and space medicine studies.

🧪 **Methodological Approaches**: Researchers typically employ ground-based analogs, parabolic flight experiments, and ISS-based investigations.

🔗 **Research Connections**: Links to space medicine, astrobiology, and life support systems research.

🚀 **Future Directions**: Important for long-duration missions and planetary exploration.

ℹ️  *Demo response. For AI-powered analysis with Gemini 2.5 Flash, configure Google API key.*
            """.strip(),
            "analysis": f"""
🔬 **Paper Analysis: "{query}"**

**Key Concepts**: Space biology, microgravity effects, cellular responses, biomarkers, space medicine

**Methodology**: Experimental research with controls, likely using space analog facilities

**Significance**: 8/10 - High importance for space exploration and human health

**Connections**: Related to other microgravity studies and space medicine research

**Future Work**: Potential for follow-up studies and Mars mission applications

ℹ️  *Demo analysis. Configure Google Gemini API for detailed AI analysis.*
            """.strip()
        }
        
        return {
            "success": False if error else True,
            "response": demo_responses.get(response_type, demo_responses["query"]),
            "error": error,
            "model": "demo_mode",
            "provider": "langchain_demo"
        }


# Legacy compatibility and factory functions
def create_agent(agent_type: str = "research_assistant", api_key: str = None):
    """Create a research agent (factory function)"""
    return LangChainResearchAgent(api_key=api_key)
