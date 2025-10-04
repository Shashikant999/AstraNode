"""Enhanced research tools for detailed paper analysis"""

import requests
import json
from typing import Dict, List, Optional
from langchain.tools import BaseTool
from langchain_core.tools import tool


@tool
def get_semantic_scholar_paper(paper_title: str) -> str:
    """
    Get detailed paper information from Semantic Scholar API
    
    Args:
        paper_title: Title of the research paper
        
    Returns:
        JSON string with detailed paper information including abstract, authors, citations
    """
    try:
        # Search for paper by title
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': paper_title,
            'limit': 1,
            'fields': 'title,authors,abstract,citationCount,referenceCount,publicationDate,journal,url,openAccessPdf'
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('data'):
            return json.dumps({"error": "Paper not found"})
        
        paper = data['data'][0]
        
        # Get citation details
        paper_id = paper.get('paperId', '')
        if paper_id:
            citations_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
            citations_params = {'fields': 'title,authors', 'limit': 10}
            citations_response = requests.get(citations_url, params=citations_params, timeout=10)
            citations_data = citations_response.json() if citations_response.ok else {}
        else:
            citations_data = {}
        
        result = {
            "title": paper.get('title', 'Unknown'),
            "authors": [author.get('name', 'Unknown') for author in paper.get('authors', [])],
            "abstract": paper.get('abstract', 'Abstract not available'),
            "publication_date": paper.get('publicationDate', 'Unknown'),
            "journal": paper.get('journal', {}).get('name', 'Unknown'),
            "citation_count": paper.get('citationCount', 0),
            "reference_count": paper.get('referenceCount', 0),
            "url": paper.get('url', ''),
            "pdf_url": paper.get('openAccessPdf', {}).get('url', '') if paper.get('openAccessPdf') else '',
            "citing_papers": [
                {
                    "title": cite.get('citingPaper', {}).get('title', 'Unknown'),
                    "authors": [a.get('name') for a in cite.get('citingPaper', {}).get('authors', [])]
                }
                for cite in citations_data.get('data', [])[:5]
            ]
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch paper details: {str(e)}"})


@tool 
def get_arxiv_paper_details(arxiv_id: str) -> str:
    """
    Get detailed information about an arXiv preprint
    
    Args:
        arxiv_id: arXiv identifier (e.g., "2301.12345")
        
    Returns:
        JSON string with paper details and full text access
    """
    try:
        import arxiv
        
        # Search for paper by ID
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results(), None)
        
        if not paper:
            return json.dumps({"error": "arXiv paper not found"})
        
        result = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "published": paper.published.isoformat() if paper.published else None,
            "updated": paper.updated.isoformat() if paper.updated else None,
            "categories": paper.categories,
            "pdf_url": paper.pdf_url,
            "entry_id": paper.entry_id,
            "comment": paper.comment,
            "journal_ref": paper.journal_ref,
            "primary_category": paper.primary_category
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch arXiv paper: {str(e)}"})


@tool
def analyze_research_collaboration_network(researcher_names: List[str]) -> str:
    """
    Analyze collaboration networks between researchers using Semantic Scholar
    
    Args:
        researcher_names: List of researcher names to analyze
        
    Returns:
        JSON string with collaboration analysis
    """
    try:
        collaboration_data = {}
        
        for researcher in researcher_names:
            # Search for author
            author_url = "https://api.semanticscholar.org/graph/v1/author/search"
            params = {'query': researcher, 'limit': 1}
            
            response = requests.get(author_url, params=params, timeout=10)
            if not response.ok:
                continue
                
            data = response.json()
            if not data.get('data'):
                continue
                
            author = data['data'][0]
            author_id = author.get('authorId')
            
            if author_id:
                # Get author's papers
                papers_url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"
                papers_params = {'fields': 'title,authors,publicationDate', 'limit': 50}
                papers_response = requests.get(papers_url, params=papers_params, timeout=10)
                
                if papers_response.ok:
                    papers_data = papers_response.json()
                    collaboration_data[researcher] = {
                        "total_papers": len(papers_data.get('data', [])),
                        "collaborators": set(),
                        "recent_papers": []
                    }
                    
                    for paper in papers_data.get('data', []):
                        # Extract collaborator names
                        for author_info in paper.get('authors', []):
                            coauthor_name = author_info.get('name', '')
                            if coauthor_name and coauthor_name != researcher:
                                collaboration_data[researcher]["collaborators"].add(coauthor_name)
                        
                        # Store recent papers
                        if len(collaboration_data[researcher]["recent_papers"]) < 5:
                            collaboration_data[researcher]["recent_papers"].append({
                                "title": paper.get('title', ''),
                                "date": paper.get('publicationDate', ''),
                                "coauthors": [a.get('name') for a in paper.get('authors', [])]
                            })
        
        # Convert sets to lists for JSON serialization
        for researcher in collaboration_data:
            collaboration_data[researcher]["collaborators"] = list(collaboration_data[researcher]["collaborators"])
        
        # Find common collaborators
        common_collabs = {}
        researchers_list = list(collaboration_data.keys())
        for i, r1 in enumerate(researchers_list):
            for r2 in researchers_list[i+1:]:
                if r1 in collaboration_data and r2 in collaboration_data:
                    common = set(collaboration_data[r1]["collaborators"]) & set(collaboration_data[r2]["collaborators"])
                    if common:
                        common_collabs[f"{r1} & {r2}"] = list(common)
        
        result = {
            "researchers_analyzed": len(collaboration_data),
            "individual_networks": collaboration_data,
            "common_collaborators": common_collabs,
            "network_insights": {
                "most_collaborative": max(collaboration_data.keys(), 
                                       key=lambda r: len(collaboration_data[r]["collaborators"])) 
                                       if collaboration_data else None,
                "total_unique_collaborators": len(set().union(*[
                    set(data["collaborators"]) for data in collaboration_data.values()
                ])) if collaboration_data else 0
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to analyze collaboration network: {str(e)}"})


# Enhanced tools list
enhanced_research_tools = [
    get_semantic_scholar_paper,
    get_arxiv_paper_details, 
    analyze_research_collaboration_network
]
