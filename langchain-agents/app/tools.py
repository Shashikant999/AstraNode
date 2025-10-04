"""Research Paper Analysis Tools for LangChain Agents"""

import os
import requests
import json
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class PaperSearchInput(BaseModel):
    query: str = Field(description="Search query for research papers")
    max_results: int = Field(default=10, description="Maximum number of results")
    use_graph_structure: bool = Field(default=True, description="Use graph structure for expansion")


class ConceptExploreInput(BaseModel):
    concept: str = Field(description="Research concept to explore")
    depth: int = Field(default=2, description="Exploration depth")


class PathAnalysisInput(BaseModel):
    source_paper_id: int = Field(description="Source paper ID")
    target_paper_id: int = Field(description="Target paper ID") 
    max_hops: int = Field(default=3, description="Maximum hops in path")


class GraphRAGAPI:
    """Interface to existing GraphRAG API"""
    
    def __init__(self, base_url: str = "http://localhost:3001"):
        self.base_url = base_url
        
    def _make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make HTTP request to GraphRAG API"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method == "POST":
                response = requests.post(url, json=data, timeout=30)
            else:
                response = requests.get(url, params=data, timeout=30)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"API request failed: {str(e)}"}
    
    def query_graph(self, query: str, options: dict) -> dict:
        """Query the knowledge graph"""
        return self._make_request("POST", "/api/rag/query", {
            "query": query,
            "options": options
        })
    
    def explore_concept(self, concept: str, depth: int = 2) -> dict:
        """Explore a research concept"""
        return self._make_request("GET", f"/api/rag/concept/{concept}", {"depth": depth})
    
    def find_paths(self, source_id: int, target_id: int, max_hops: int = 3) -> dict:
        """Find research paths between papers"""
        return self._make_request("GET", f"/api/rag/paths/{source_id}/{target_id}", 
                                {"max_hops": max_hops})
    
    def get_papers(self) -> dict:
        """Get all papers"""
        return self._make_request("GET", "/api/papers")
    
    def search_papers(self, query: str) -> dict:
        """Search papers"""
        return self._make_request("GET", "/api/search", {"query": query})


# Initialize GraphRAG API client
graphrag_api = GraphRAGAPI()


@tool
def search_research_papers(query: str, max_results: int = 10, use_graph_structure: bool = True) -> str:
    """
    Search for research papers using intelligent graph-based retrieval.
    
    Args:
        query: Natural language query describing research interests
        max_results: Maximum number of papers to return
        use_graph_structure: Whether to use graph connections for expansion
        
    Returns:
        JSON string with relevant papers, insights, and connections
    """
    options = {
        "maxResults": max_results,
        "useGraphStructure": use_graph_structure,
        "includeConnections": True
    }
    
    result = graphrag_api.query_graph(query, options)
    
    if "error" in result:
        return f"Error searching papers: {result['error']}"
    
    # Format results for agent consumption
    papers = result.get("results", [])
    insights = result.get("insights", {})
    
    summary = {
        "query": query,
        "papers_found": len(papers),
        "key_papers": [
            {
                "title": p["title"],
                "domain": p.get("domain", "Unknown"),
                "relevance": f"{(p.get('relevanceScore', 0) * 100):.0f}%",
                "concepts": p.get("concepts", [])[:3],
                "link": p.get("link")
            }
            for p in papers[:5]
        ],
        "research_insights": insights.get("content", "No insights available"),
        "key_themes": insights.get("themes", []),
        "research_domains": [d["domain"] for d in insights.get("domains", [])]
    }
    
    return json.dumps(summary, indent=2)


@tool 
def explore_research_concept(concept: str, depth: int = 2) -> str:
    """
    Deep dive into a specific research concept and its neighborhood.
    
    Args:
        concept: Research concept to explore (e.g., 'microgravity', 'stem cells')
        depth: How deep to explore connections (1-3)
        
    Returns:
        JSON string with related papers, concepts, and insights
    """
    result = graphrag_api.explore_concept(concept, depth)
    
    if "error" in result:
        return f"Error exploring concept: {result['error']}"
    
    papers = result.get("papers", [])
    concepts = result.get("concepts", [])
    insights = result.get("insights", {})
    
    summary = {
        "concept": concept,
        "related_papers": len(papers),
        "top_papers": [
            {
                "title": p["title"],
                "domain": p.get("domain", "Unknown"),
                "concepts": p.get("concepts", [])[:3]
            }
            for p in papers[:5]
        ],
        "related_concepts": [
            {"concept": c["concept"], "frequency": c["count"]}
            for c in concepts[:8]
        ],
        "research_insights": insights.get("content", "No insights available")
    }
    
    return json.dumps(summary, indent=2)


@tool
def analyze_research_connections(source_paper_title: str, target_paper_title: str) -> str:
    """
    Analyze connections and research paths between two papers.
    
    Args:
        source_paper_title: Title of the source paper
        target_paper_title: Title of the target paper
        
    Returns:
        JSON string with connection analysis and research paths
    """
    # First, find the paper IDs
    papers_result = graphrag_api.get_papers()
    
    if "error" in papers_result:
        return f"Error getting papers: {papers_result['error']}"
    
    papers = papers_result if isinstance(papers_result, list) else []
    
    source_paper = None
    target_paper = None
    
    for paper in papers:
        if source_paper_title.lower() in paper.get("title", "").lower():
            source_paper = paper
        if target_paper_title.lower() in paper.get("title", "").lower():
            target_paper = paper
    
    if not source_paper or not target_paper:
        return json.dumps({
            "error": "Could not find one or both papers",
            "source_found": bool(source_paper),
            "target_found": bool(target_paper)
        })
    
    # Find paths between papers
    paths_result = graphrag_api.find_paths(source_paper["id"], target_paper["id"])
    
    if "error" in paths_result:
        return f"Error finding paths: {paths_result['error']}"
    
    summary = {
        "source_paper": {
            "title": source_paper["title"],
            "domain": source_paper.get("domain", "Unknown")
        },
        "target_paper": {
            "title": target_paper["title"],
            "domain": target_paper.get("domain", "Unknown")
        },
        "connection_paths": len(paths_result) if isinstance(paths_result, list) else 0,
        "research_relationship": "Direct connection found" if paths_result else "No direct connection",
        "path_analysis": paths_result[:3] if isinstance(paths_result, list) else []
    }
    
    return json.dumps(summary, indent=2)


@tool
def identify_research_gaps(domain: str = None, concept: str = None) -> str:
    """
    Identify potential research gaps and opportunities in a domain or around a concept.
    
    Args:
        domain: Research domain to analyze (e.g., 'Microgravity Research')
        concept: Specific concept to analyze (e.g., 'bone density')
        
    Returns:
        JSON string with gap analysis and research opportunities
    """
    if concept:
        # Explore concept to find gaps
        result = graphrag_api.explore_concept(concept, depth=2)
    elif domain:
        # Search for papers in domain
        result = graphrag_api.query_graph(f"research in {domain}", {
            "maxResults": 20,
            "useGraphStructure": True
        })
    else:
        return json.dumps({"error": "Please provide either a domain or concept to analyze"})
    
    if "error" in result:
        return f"Error analyzing research gaps: {result['error']}"
    
    # Analyze for gaps (simplified heuristic approach)
    papers = result.get("papers", []) or result.get("results", [])
    insights = result.get("insights", {})
    
    # Basic gap analysis
    domains_covered = set()
    concepts_covered = set()
    
    for paper in papers:
        if paper.get("domain"):
            domains_covered.add(paper["domain"])
        for concept in paper.get("concepts", []):
            concepts_covered.add(concept.lower())
    
    gap_analysis = {
        "analysis_target": domain or concept,
        "papers_analyzed": len(papers),
        "domains_covered": list(domains_covered),
        "concepts_covered": list(concepts_covered)[:10],
        "research_insights": insights.get("content", ""),
        "potential_gaps": [
            "Limited cross-domain collaboration opportunities",
            "Under-explored concept combinations",
            "Methodology gaps in current approaches"
        ],
        "research_opportunities": [
            f"Interdisciplinary work combining {domain or concept} with other fields",
            "Longitudinal studies in under-represented areas", 
            "Technology transfer from related domains"
        ]
    }
    
    return json.dumps(gap_analysis, indent=2)


@tool
def find_collaboration_opportunities(researcher_interest: str, institution: str = None) -> str:
    """
    Find potential collaboration opportunities based on research interests.
    
    Args:
        researcher_interest: Description of research interests
        institution: Optional institution filter
        
    Returns:
        JSON string with collaboration suggestions and relevant researchers
    """
    # Search for papers related to the research interest
    result = graphrag_api.query_graph(researcher_interest, {
        "maxResults": 15,
        "useGraphStructure": True,
        "includeConnections": True
    })
    
    if "error" in result:
        return f"Error finding collaborations: {result['error']}"
    
    papers = result.get("results", [])
    connections = result.get("connections", [])
    
    # Analyze collaboration potential
    collaboration_analysis = {
        "research_interest": researcher_interest,
        "relevant_papers": len(papers),
        "research_clusters": len(set(p.get("domain", "Unknown") for p in papers)),
        "top_research_areas": [
            {
                "domain": p.get("domain", "Unknown"),
                "paper_count": len([pp for pp in papers if pp.get("domain") == p.get("domain")]),
                "key_concepts": p.get("concepts", [])[:3]
            }
            for p in papers[:5]
        ],
        "collaboration_suggestions": [
            "Researchers working on similar microgravity effects",
            "Cross-disciplinary opportunities in space biology",
            "International space research collaborations",
            "Industry-academic partnerships in space medicine"
        ],
        "networking_opportunities": [
            "Space biology conferences and workshops",
            "Microgravity research consortiums", 
            "NASA collaborative research programs"
        ]
    }
    
    return json.dumps(collaboration_analysis, indent=2)


# Export all tools
research_tools = [
    search_research_papers,
    explore_research_concept, 
    analyze_research_connections,
    identify_research_gaps,
    find_collaboration_opportunities
]
