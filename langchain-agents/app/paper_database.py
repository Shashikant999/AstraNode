"""
Space Biology Paper Database - CSV Integration
Loads and searches the 607 space biology papers from SB_publication_PMC.csv
"""

import csv
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paper:
    """Represents a research paper from the database"""
    title: str
    link: str
    pmc_id: str = ""
    
    def __post_init__(self):
        """Extract PMC ID from the link"""
        if "pmc/articles/" in self.link:
            # Extract PMC ID from URL like: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4136787/
            match = re.search(r'PMC(\d+)', self.link)
            self.pmc_id = match.group(0) if match else ""


class SpaceBiologyPaperDB:
    """Database of 607 space biology research papers"""
    
    def __init__(self, csv_path: str = None):
        """Initialize with CSV file path"""
        if csv_path is None:
            # Auto-detect CSV path - works on any system
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            csv_path = os.path.join(project_root, "SB_publication_PMC.csv")
            
            # Fallback paths for different project structures
            fallback_paths = [
                os.path.join(current_dir, "..", "..", "SB_publication_PMC.csv"),
                os.path.join(current_dir, "SB_publication_PMC.csv"),
                "SB_publication_PMC.csv"
            ]
            
            if not os.path.exists(csv_path):
                for fallback in fallback_paths:
                    if os.path.exists(fallback):
                        csv_path = fallback
                        break
        
        self.csv_path = csv_path
        self.papers: List[Paper] = []
        self._load_papers()
    
    def _load_papers(self):
        """Load papers from CSV file"""
        try:
            with open(self.csv_path, 'r', encoding='utf-8-sig') as file:  # Handle BOM
                reader = csv.DictReader(file)
                for row in reader:
                    # Handle potential BOM in column names
                    title_key = 'Title' if 'Title' in row else list(row.keys())[0]
                    link_key = 'Link' if 'Link' in row else list(row.keys())[1]
                    
                    paper = Paper(
                        title=row[title_key].strip(),
                        link=row[link_key].strip()
                    )
                    self.papers.append(paper)
            
            print(f"✅ Loaded {len(self.papers)} papers from space biology database")
        except FileNotFoundError:
            print(f"❌ CSV file not found: {self.csv_path}")
            self.papers = []
        except Exception as e:
            print(f"❌ Error loading papers: {e}")
            self.papers = []
    
    def search_papers(self, query: str, max_results: int = 20) -> List[Paper]:
        """Search papers by title keywords"""
        query_terms = query.lower().split()
        matching_papers = []
        
        for paper in self.papers:
            title_lower = paper.title.lower()
            # Calculate relevance score based on matching terms
            score = sum(1 for term in query_terms if term in title_lower)
            
            if score > 0:
                matching_papers.append((paper, score))
        
        # Sort by relevance score (descending)
        matching_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches
        return [paper for paper, score in matching_papers[:max_results]]
    
    def get_papers_by_keywords(self, keywords: List[str]) -> List[Paper]:
        """Get papers containing any of the specified keywords"""
        matching_papers = []
        
        for paper in self.papers:
            title_lower = paper.title.lower()
            if any(keyword.lower() in title_lower for keyword in keywords):
                matching_papers.append(paper)
        
        return matching_papers
    
    def get_random_sample(self, n: int = 10) -> List[Paper]:
        """Get a random sample of papers"""
        import random
        return random.sample(self.papers, min(n, len(self.papers)))
    
    def get_paper_count(self) -> int:
        """Get total number of papers in database"""
        return len(self.papers)
    
    def get_papers_by_topic(self, topic: str) -> Dict[str, Any]:
        """Analyze papers related to a specific topic"""
        relevant_papers = self.search_papers(topic, max_results=50)
        
        # Categorize papers
        categories = {
            'microgravity': [],
            'radiation': [],
            'muscle': [],
            'bone': [],
            'cell': [],
            'gene': [],
            'protein': [],
            'other': []
        }
        
        for paper in relevant_papers:
            title_lower = paper.title.lower()
            categorized = False
            
            for category, keywords in {
                'microgravity': ['microgravity', 'gravity', 'spaceflight', 'space'],
                'radiation': ['radiation', 'cosmic', 'ion', 'dose'],
                'muscle': ['muscle', 'skeletal', 'atrophy'],
                'bone': ['bone', 'osteo', 'calcium'],
                'cell': ['cell', 'cellular', 'stem'],
                'gene': ['gene', 'expression', 'transcr'],
                'protein': ['protein', 'enzyme', 'kinase']
            }.items():
                if any(keyword in title_lower for keyword in keywords):
                    categories[category].append(paper)
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(paper)
        
        return {
            'topic': topic,
            'total_found': len(relevant_papers),
            'categories': categories,
            'top_papers': relevant_papers[:10]
        }


# Global instance for easy access
_paper_db = None

def get_paper_database() -> SpaceBiologyPaperDB:
    """Get or create the global paper database instance"""
    global _paper_db
    if _paper_db is None:
        _paper_db = SpaceBiologyPaperDB()
    return _paper_db


def search_research_papers(query: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """Search function for integration with agents"""
    db = get_paper_database()
    papers = db.search_papers(query, max_results)
    
    return [
        {
            'title': paper.title,
            'link': paper.link,
            'pmc_id': paper.pmc_id,
            'source': 'Space Biology Database'
        }
        for paper in papers
    ]


def get_topic_analysis(topic: str) -> Dict[str, Any]:
    """Get comprehensive topic analysis from paper database"""
    db = get_paper_database()
    return db.get_papers_by_topic(topic)


def get_database_stats() -> Dict[str, Any]:
    """Get statistics about the paper database"""
    db = get_paper_database()
    
    # Sample analysis
    sample_papers = db.get_random_sample(100)
    topic_counts = {}
    
    for paper in sample_papers:
        title_words = paper.title.lower().split()
        for word in ['microgravity', 'radiation', 'muscle', 'bone', 'cell', 'gene']:
            if word in ' '.join(title_words):
                topic_counts[word] = topic_counts.get(word, 0) + 1
    
    return {
        'total_papers': db.get_paper_count(),
        'database_source': 'SB_publication_PMC.csv',
        'topic_distribution': topic_counts,
        'sample_titles': [p.title for p in sample_papers[:5]]
    }


if __name__ == "__main__":
    # Test the database
    print("Testing Space Biology Paper Database...")
    db = get_paper_database()
    print(f"Loaded {db.get_paper_count()} papers")
    
    # Test search
    results = db.search_papers("microgravity muscle", max_results=5)
    print(f"\nFound {len(results)} papers for 'microgravity muscle':")
    for paper in results:
        print(f"- {paper.title}")
        print(f"  {paper.link}")
    
    # Test topic analysis
    analysis = db.get_papers_by_topic("bone")
    print(f"\nBone research: {analysis['total_found']} papers found")
    print(f"Categories: {[k for k, v in analysis['categories'].items() if v]}")
