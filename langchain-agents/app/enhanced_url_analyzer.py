"""
Enhanced GraphRAG with URL Content Analysis
Fetches content from research paper URLs and finds interlinked topics
"""

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from typing import List, Dict, Any, Optional, Set
import re
from dataclasses import dataclass
import json
from .gemini_agent import GeminiResearchAgent


@dataclass
class PaperContent:
    """Represents extracted content from a research paper URL"""
    title: str
    abstract: str
    full_text: str
    keywords: List[str]
    url: str
    pmc_id: str = ""
    
    def get_summary(self) -> str:
        """Get a summary of the paper content"""
        content = f"Title: {self.title}\n"
        if self.abstract:
            content += f"Abstract: {self.abstract[:500]}...\n"
        if self.keywords:
            content += f"Keywords: {', '.join(self.keywords)}\n"
        return content


class URLContentAnalyzer:
    """Analyzes content from research paper URLs"""
    
    def __init__(self, gemini_agent: GeminiResearchAgent):
        self.gemini_agent = gemini_agent
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def fetch_paper_content(self, url: str) -> Optional[PaperContent]:
        """Fetch and extract content from a research paper URL"""
        try:
            # Handle PMC URLs specifically
            if "pmc/articles/PMC" in url:
                return self._fetch_pmc_content(url)
            else:
                return self._fetch_generic_content(url)
                
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            return None
    
    def _fetch_pmc_content(self, url: str) -> Optional[PaperContent]:
        """Fetch content from PMC (PubMed Central) URLs"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ""
            title_elem = soup.find('h1', class_='content-title') or soup.find('title')
            if title_elem:
                title = title_elem.get_text().strip()
            
            # Extract abstract
            abstract = ""
            abstract_elem = soup.find('div', class_='abstract') or soup.find('div', id='abstract')
            if abstract_elem:
                abstract = abstract_elem.get_text().strip()
            
            # Extract keywords
            keywords = []
            keyword_elem = soup.find('div', class_='kwd-group') or soup.find('div', class_='keywords')
            if keyword_elem:
                keyword_texts = keyword_elem.find_all(['span', 'a'])
                keywords = [kw.get_text().strip() for kw in keyword_texts if kw.get_text().strip()]
            
            # Extract main content
            full_text = ""
            content_elem = soup.find('div', class_='tsec') or soup.find('div', class_='article-content')
            if content_elem:
                full_text = content_elem.get_text().strip()[:5000]  # Limit to 5000 chars
            
            # Extract PMC ID
            pmc_id = ""
            pmc_match = re.search(r'PMC(\d+)', url)
            if pmc_match:
                pmc_id = pmc_match.group(0)
            
            return PaperContent(
                title=title,
                abstract=abstract,
                full_text=full_text,
                keywords=keywords,
                url=url,
                pmc_id=pmc_id
            )
            
        except Exception as e:
            print(f"Error fetching PMC content: {e}")
            return None
    
    def _fetch_generic_content(self, url: str) -> Optional[PaperContent]:
        """Fetch content from generic research paper URLs"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ""
            title_elem = soup.find('title') or soup.find('h1')
            if title_elem:
                title = title_elem.get_text().strip()
            
            # Try to extract abstract/summary
            abstract = ""
            abstract_selectors = [
                'div[class*="abstract"]',
                'div[id*="abstract"]',
                'section[class*="abstract"]',
                'p[class*="summary"]'
            ]
            
            for selector in abstract_selectors:
                abstract_elem = soup.select_one(selector)
                if abstract_elem:
                    abstract = abstract_elem.get_text().strip()
                    break
            
            # Extract main text content
            full_text = soup.get_text()[:3000]  # Limit to 3000 chars
            
            return PaperContent(
                title=title,
                abstract=abstract,
                full_text=full_text,
                keywords=[],
                url=url
            )
            
        except Exception as e:
            print(f"Error fetching generic content: {e}")
            return None


class EnhancedGraphRAGAnalyzer:
    """Enhanced GraphRAG that analyzes URL content and finds interlinked topics"""
    
    def __init__(self, gemini_agent: GeminiResearchAgent):
        self.gemini_agent = gemini_agent
        self.url_analyzer = URLContentAnalyzer(gemini_agent)
        self.content_cache = {}  # Cache for fetched content
    
    def analyze_papers_for_prompt(self, papers: List[Dict], user_prompt: str, max_papers: int = 10) -> Dict[str, Any]:
        """
        Analyze paper URLs to find topics related to user prompt
        
        Args:
            papers: List of paper dictionaries with 'title' and 'link'
            user_prompt: User's research query
            max_papers: Maximum number of papers to analyze
        
        Returns:
            Dictionary with analysis results and interlinked topics
        """
        print(f"🔍 Analyzing {min(len(papers), max_papers)} papers for prompt: '{user_prompt}'")
        
        # Fetch content from paper URLs
        paper_contents = []
        for i, paper in enumerate(papers[:max_papers]):
            print(f"📄 Fetching content {i+1}/{min(len(papers), max_papers)}: {paper.get('title', 'Unknown')}")
            
            url = paper.get('link', '')
            if not url:
                continue
                
            # Check cache first
            if url in self.content_cache:
                content = self.content_cache[url]
            else:
                content = self.url_analyzer.fetch_paper_content(url)
                if content:
                    self.content_cache[url] = content
                    time.sleep(1)  # Rate limiting
            
            if content:
                paper_contents.append(content)
        
        print(f"✅ Successfully fetched content from {len(paper_contents)} papers")
        
        # Use Gemini to analyze content and find interlinked topics
        return self._analyze_content_with_gemini(paper_contents, user_prompt)
    
    def _analyze_content_with_gemini(self, paper_contents: List[PaperContent], user_prompt: str) -> Dict[str, Any]:
        """Use Gemini 2.5 Pro to analyze content and find interlinked topics"""
        
        if not paper_contents:
            return {
                'success': False,
                'error': 'No paper content could be fetched',
                'interlinked_topics': [],
                'analysis': 'Unable to analyze papers - content not accessible'
            }
        
        # Prepare content summary for Gemini
        content_summaries = []
        for i, content in enumerate(paper_contents, 1):
            summary = f"""
Paper {i}:
Title: {content.title}
URL: {content.url}
Abstract: {content.abstract[:300]}...
Keywords: {', '.join(content.keywords) if content.keywords else 'None provided'}
Content Preview: {content.full_text[:400]}...
---
"""
            content_summaries.append(summary)
        
        # Create comprehensive analysis prompt
        analysis_prompt = f"""
You are an expert research analyst with access to {len(paper_contents)} research papers. 
Your task is to analyze these papers and identify interlinked topics specifically related to this user query:

USER QUERY: "{user_prompt}"

RESEARCH PAPERS CONTENT:
{chr(10).join(content_summaries)}

Please provide a comprehensive analysis with the following structure:

1. TOPIC INTERCONNECTIONS:
   - Identify 5-8 key topics that appear across multiple papers
   - Show how these topics connect to the user's query
   - Explain the relationships between topics

2. CROSS-PAPER INSIGHTS:
   - Find patterns, trends, or contradictions across papers
   - Identify research gaps or opportunities
   - Highlight novel connections between different studies

3. QUERY-SPECIFIC ANALYSIS:
   - Direct relevance to the user's query
   - Key findings that address the query
   - Recommended research directions

4. INTERLINKED RESEARCH NETWORK:
   - Create a conceptual map of how papers relate to each other
   - Identify central themes and peripheral topics
   - Show research collaboration opportunities

5. ACTIONABLE INSIGHTS:
   - Practical applications of the research
   - Future research directions
   - Policy or clinical implications

Please provide detailed analysis with specific paper references and clear explanations of topic interconnections.
"""
        
        try:
            if not self.gemini_agent.api_working:
                return self._generate_demo_analysis(paper_contents, user_prompt)
            
            # Use Gemini 2.5 Pro thinking capabilities
            response = self.gemini_agent.model.generate_content(analysis_prompt)
            
            # Extract topics and relationships
            interlinked_topics = self._extract_topics_from_analysis(response.text)
            
            return {
                'success': True,
                'analysis': response.text,
                'interlinked_topics': interlinked_topics,
                'papers_analyzed': len(paper_contents),
                'user_query': user_prompt,
                'model': 'gemini-2.5-pro-thinking',
                'analysis_type': 'url_content_analysis'
            }
            
        except Exception as e:
            print(f"Error in Gemini analysis: {e}")
            return self._generate_demo_analysis(paper_contents, user_prompt)
    
    def _extract_topics_from_analysis(self, analysis_text: str) -> List[Dict[str, Any]]:
        """Extract structured topics from Gemini analysis"""
        # Use regex to find topic sections and extract key information
        topics = []
        
        # Look for numbered topics, bullet points, or header patterns
        topic_patterns = [
            r'(?:Topic|Theme|Area)\s*\d+[:\.]?\s*([^.\n]+)',
            r'(?:•|\*|-)\s*([^.\n]+)',
            r'(?:Key finding|Main topic|Central theme):\s*([^.\n]+)'
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, analysis_text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 10:  # Filter out very short matches
                    topics.append({
                        'topic': match.strip(),
                        'relevance': 'high',  # Could be enhanced with sentiment analysis
                        'connections': []  # Could be enhanced with relationship extraction
                    })
        
        # Return unique topics (first 10)
        unique_topics = []
        seen = set()
        for topic in topics:
            topic_lower = topic['topic'].lower()
            if topic_lower not in seen and len(unique_topics) < 10:
                seen.add(topic_lower)
                unique_topics.append(topic)
        
        return unique_topics
    
    def _generate_demo_analysis(self, paper_contents: List[PaperContent], user_prompt: str) -> Dict[str, Any]:
        """Generate demo analysis when Gemini API is not available"""
        
        # Extract keywords and common themes
        all_keywords = []
        all_titles = []
        
        for content in paper_contents:
            all_keywords.extend(content.keywords)
            all_titles.append(content.title)
        
        # Simple keyword frequency analysis
        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # Top keywords as interlinked topics
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:8]
        
        interlinked_topics = []
        for keyword, freq in top_keywords:
            interlinked_topics.append({
                'topic': keyword,
                'relevance': 'high' if freq > 2 else 'medium',
                'frequency': freq,
                'connections': []
            })
        
        demo_analysis = f"""
DEMO MODE - URL Content Analysis for: "{user_prompt}"

📊 PAPERS ANALYZED: {len(paper_contents)} research papers
🔗 CONTENT SOURCES: Successfully fetched from {len([c for c in paper_contents if c.abstract])} abstracts

🎯 KEY INTERLINKED TOPICS:
{chr(10).join([f"• {topic['topic']} (mentioned {topic['frequency']} times)" for topic in interlinked_topics[:5]])}

📋 PAPER TITLES ANALYZED:
{chr(10).join([f"• {title}" for title in all_titles[:10]])}

💡 ANALYSIS INSIGHTS:
- Cross-referenced {len(all_keywords)} total keywords across papers
- Identified {len(set(all_keywords))} unique research concepts  
- Found {len(top_keywords)} recurring themes relevant to your query

🔄 TOPIC INTERCONNECTIONS:
The analyzed papers show connections through shared keywords and research methodologies.
Common themes relate to your query "{user_prompt}" through overlapping research domains.

⚠️  Note: This is a demo analysis. Enable Gemini API for comprehensive AI-powered analysis with deep topic interconnections and research insights.
"""
        
        return {
            'success': True,
            'analysis': demo_analysis,
            'interlinked_topics': interlinked_topics,
            'papers_analyzed': len(paper_contents),
            'user_query': user_prompt,
            'model': 'demo_url_analyzer',
            'analysis_type': 'demo_url_content_analysis'
        }


def create_enhanced_graphrag_analyzer():
    """Factory function to create enhanced GraphRAG analyzer"""
    from .gemini_agent import GeminiResearchAgent
    
    try:
        gemini_agent = GeminiResearchAgent()
        return EnhancedGraphRAGAnalyzer(gemini_agent)
    except Exception as e:
        print(f"Failed to create enhanced analyzer: {e}")
        return None
