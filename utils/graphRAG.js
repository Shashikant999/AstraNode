const OpenAI = require('openai');

class GraphRAG {
  constructor() {
    this.openai = process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY !== 'demo_mode' 
      ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
      : null;
    
    this.graphData = null;
    this.papers = [];
    this.embeddings = new Map(); // Cache for paper embeddings
    this.conceptEmbeddings = new Map(); // Cache for concept embeddings
  }

  setGraphData(graphData, papers) {
    this.graphData = graphData;
    this.papers = papers;
  }

  // Enhanced query processing with graph context
  async queryGraph(query, options = {}) {
    const {
      maxResults = 10,
      includeConnections = true,
      semanticThreshold = 0.7,
      useGraphStructure = true
    } = options;

    console.log(`GraphRAG Query: "${query}"`);

    try {
      // Step 1: Find relevant papers using semantic search
      const relevantPapers = await this.semanticSearch(query, maxResults * 2);
      
      // Step 2: Expand results using graph structure
      const expandedResults = useGraphStructure 
        ? this.expandWithGraphContext(relevantPapers, query)
        : relevantPapers;

      // Step 3: Generate AI-powered insights
      const insights = await this.generateInsights(query, expandedResults);

      // Step 4: Find connection paths between relevant papers
      const connections = includeConnections 
        ? this.findRelevantConnections(expandedResults)
        : [];

      return {
        query,
        results: expandedResults.slice(0, maxResults),
        insights,
        connections,
        graphContext: this.getGraphContext(expandedResults),
        metadata: {
          totalRelevant: expandedResults.length,
          semanticMatches: relevantPapers.length,
          graphExpanded: expandedResults.length - relevantPapers.length
        }
      };

    } catch (error) {
      console.error('GraphRAG query error:', error);
      return this.fallbackQuery(query, maxResults);
    }
  }

  // Semantic search using embeddings or fallback to keyword matching
  async semanticSearch(query, maxResults) {
    if (this.openai) {
      return await this.embeddingBasedSearch(query, maxResults);
    } else {
      return this.keywordBasedSearch(query, maxResults);
    }
  }

  async embeddingBasedSearch(query, maxResults) {
    try {
      // Get query embedding
      const queryEmbedding = await this.getEmbedding(query);
      
      // Get or create paper embeddings
      const paperScores = [];
      
      for (const paper of this.papers) {
        const paperText = `${paper.title} ${paper.concepts?.join(' ') || ''} ${paper.domain || ''}`;
        const paperEmbedding = await this.getEmbedding(paperText);
        
        const similarity = this.cosineSimilarity(queryEmbedding, paperEmbedding);
        paperScores.push({ paper, similarity });
      }

      return paperScores
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, maxResults)
        .filter(item => item.similarity > 0.5)
        .map(item => ({ ...item.paper, relevanceScore: item.similarity }));

    } catch (error) {
      console.error('Embedding search failed:', error);
      return this.keywordBasedSearch(query, maxResults);
    }
  }

  keywordBasedSearch(query, maxResults) {
    const queryTerms = query.toLowerCase().split(/\s+/).filter(term => term.length > 2);
    const paperScores = [];

    for (const paper of this.papers) {
      let score = 0;
      const paperText = `${paper.title} ${paper.concepts?.join(' ') || ''} ${paper.domain || ''}`.toLowerCase();
      
      queryTerms.forEach(term => {
        // Exact matches
        const exactMatches = (paperText.match(new RegExp(term, 'g')) || []).length;
        score += exactMatches * 2;
        
        // Partial matches
        if (paperText.includes(term)) {
          score += 1;
        }
        
        // Concept matches (higher weight)
        if (paper.concepts?.some(concept => concept.toLowerCase().includes(term))) {
          score += 3;
        }
      });

      if (score > 0) {
        paperScores.push({ ...paper, relevanceScore: score });
      }
    }

    return paperScores
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, maxResults);
  }

  // Expand results using graph neighborhood
  expandWithGraphContext(papers, query) {
    if (!this.graphData) return papers;

    const paperIds = new Set(papers.map(p => p.id));
    const expandedIds = new Set(paperIds);
    
    // Add connected papers with high relevance
    papers.forEach(paper => {
      const connections = this.graphData.links.filter(link => 
        link.source === paper.id || link.target === paper.id
      );
      
      connections.forEach(connection => {
        const connectedId = connection.source === paper.id ? connection.target : connection.source;
        
        if (!expandedIds.has(connectedId) && connection.strength > 0.6) {
          const connectedPaper = this.papers.find(p => p.id === connectedId);
          if (connectedPaper) {
            expandedIds.add(connectedId);
            papers.push({
              ...connectedPaper,
              relevanceScore: (papers.find(p => p.id === paper.id)?.relevanceScore || 0) * connection.strength,
              connectionReason: `Connected to "${paper.title}" (${(connection.strength * 100).toFixed(0)}% similarity)`
            });
          }
        }
      });
    });

    return papers.sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));
  }

  // Generate AI insights about the query results
  async generateInsights(query, papers) {
    if (!this.openai || papers.length === 0) {
      return this.generateBasicInsights(query, papers);
    }

    try {
      const paperSummary = papers.slice(0, 5).map(p => 
        `- "${p.title}" (Domain: ${p.domain}, Concepts: ${p.concepts?.slice(0, 3).join(', ') || 'N/A'})`
      ).join('\n');

      const prompt = `
      Based on the following research papers and query, provide insights about the research landscape:
      
      Query: "${query}"
      
      Relevant Papers:
      ${paperSummary}
      
      Please provide:
      1. Key research themes and patterns
      2. Potential research gaps or opportunities  
      3. Important connections between the papers
      4. Suggestions for further exploration
      
      Keep the response concise and actionable.
      `;

      const response = await this.openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.7,
        max_tokens: 500
      });

      return {
        aiGenerated: true,
        content: response.choices[0].message.content.trim(),
        themes: this.extractThemes(papers),
        domains: this.analyzeDomainDistribution(papers)
      };

    } catch (error) {
      console.error('AI insight generation failed:', error);
      return this.generateBasicInsights(query, papers);
    }
  }

  generateBasicInsights(query, papers) {
    const themes = this.extractThemes(papers);
    const domains = this.analyzeDomainDistribution(papers);
    
    const insights = [];
    
    if (domains.length > 1) {
      insights.push(`This query spans multiple research domains: ${domains.slice(0, 3).map(d => d.domain).join(', ')}`);
    }
    
    if (themes.length > 0) {
      insights.push(`Key research themes include: ${themes.slice(0, 5).join(', ')}`);
    }
    
    const avgConnections = papers.reduce((sum, p) => sum + (p.degree || 0), 0) / papers.length;
    if (avgConnections > 5) {
      insights.push('These papers are highly interconnected, suggesting a mature research area');
    } else if (avgConnections < 2) {
      insights.push('These papers have few connections, suggesting emerging or niche research areas');
    }

    return {
      aiGenerated: false,
      content: insights.join('. ') + '.',
      themes,
      domains
    };
  }

  // Find relevant connections between papers
  findRelevantConnections(papers) {
    if (!this.graphData) return [];

    const paperIds = new Set(papers.map(p => p.id));
    
    return this.graphData.links
      .filter(link => paperIds.has(link.source) && paperIds.has(link.target))
      .map(link => ({
        ...link,
        sourcePaper: papers.find(p => p.id === link.source),
        targetPaper: papers.find(p => p.id === link.target)
      }))
      .sort((a, b) => b.strength - a.strength)
      .slice(0, 10);
  }

  // Get graph context for the results
  getGraphContext(papers) {
    if (!this.graphData) return null;

    const paperIds = new Set(papers.map(p => p.id));
    
    // Find subgraph
    const relevantNodes = this.graphData.nodes.filter(node => paperIds.has(node.id));
    const relevantLinks = this.graphData.links.filter(link => 
      paperIds.has(link.source) && paperIds.has(link.target)
    );

    return {
      nodes: relevantNodes,
      links: relevantLinks,
      stats: {
        nodeCount: relevantNodes.length,
        linkCount: relevantLinks.length,
        density: relevantLinks.length / (relevantNodes.length * (relevantNodes.length - 1) / 2)
      }
    };
  }

  // Multi-hop reasoning - find papers connected through paths
  async findResearchPaths(startPaperId, endPaperId, maxHops = 3) {
    if (!this.graphData) return [];

    const paths = [];
    const visited = new Set();
    
    const dfs = (currentId, targetId, path, hops) => {
      if (hops > maxHops) return;
      if (currentId === targetId && path.length > 1) {
        paths.push([...path]);
        return;
      }
      
      visited.add(currentId);
      
      const connections = this.graphData.links.filter(link => 
        link.source === currentId || link.target === currentId
      );
      
      connections.forEach(connection => {
        const nextId = connection.source === currentId ? connection.target : connection.source;
        
        if (!visited.has(nextId)) {
          path.push({
            paperId: nextId,
            connection: connection,
            paper: this.papers.find(p => p.id === nextId)
          });
          
          dfs(nextId, targetId, path, hops + 1);
          path.pop();
        }
      });
      
      visited.delete(currentId);
    };

    dfs(startPaperId, endPaperId, [{ paperId: startPaperId }], 0);
    
    return paths.sort((a, b) => {
      const strengthA = a.reduce((sum, node) => sum + (node.connection?.strength || 0), 0);
      const strengthB = b.reduce((sum, node) => sum + (node.connection?.strength || 0), 0);
      return strengthB - strengthA;
    }).slice(0, 5);
  }

  // Concept-based exploration
  async exploreConceptNeighborhood(concept, depth = 2) {
    const relatedPapers = this.papers.filter(paper => 
      paper.concepts?.some(c => c.toLowerCase().includes(concept.toLowerCase()))
    );

    if (relatedPapers.length === 0) return { papers: [], concepts: [], insights: null };

    // Find related concepts
    const relatedConcepts = new Map();
    relatedPapers.forEach(paper => {
      paper.concepts?.forEach(c => {
        if (c.toLowerCase() !== concept.toLowerCase()) {
          relatedConcepts.set(c, (relatedConcepts.get(c) || 0) + 1);
        }
      });
    });

    // Expand to connected papers
    const expandedPapers = depth > 1 ? this.expandWithGraphContext(relatedPapers, concept) : relatedPapers;

    return {
      papers: expandedPapers.slice(0, 20),
      concepts: [...relatedConcepts.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([concept, count]) => ({ concept, count })),
      insights: await this.generateInsights(`Research related to ${concept}`, expandedPapers)
    };
  }

  // Helper methods
  async getEmbedding(text) {
    if (!this.openai) return null;

    const cacheKey = text.substring(0, 100);
    if (this.embeddings.has(cacheKey)) {
      return this.embeddings.get(cacheKey);
    }

    try {
      const response = await this.openai.embeddings.create({
        model: "text-embedding-ada-002",
        input: text
      });

      const embedding = response.data[0].embedding;
      this.embeddings.set(cacheKey, embedding);
      return embedding;

    } catch (error) {
      console.error('Embedding generation failed:', error);
      return null;
    }
  }

  cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB) return 0;

    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));

    return dotProduct / (magnitudeA * magnitudeB) || 0;
  }

  extractThemes(papers) {
    const conceptCounts = new Map();
    
    papers.forEach(paper => {
      paper.concepts?.forEach(concept => {
        conceptCounts.set(concept, (conceptCounts.get(concept) || 0) + 1);
      });
    });

    return [...conceptCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([concept]) => concept);
  }

  analyzeDomainDistribution(papers) {
    const domainCounts = new Map();
    
    papers.forEach(paper => {
      const domain = paper.domain || 'Unknown';
      domainCounts.set(domain, (domainCounts.get(domain) || 0) + 1);
    });

    return [...domainCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([domain, count]) => ({ domain, count }));
  }

  fallbackQuery(query, maxResults) {
    const results = this.keywordBasedSearch(query, maxResults);
    
    return {
      query,
      results,
      insights: this.generateBasicInsights(query, results),
      connections: [],
      graphContext: null,
      metadata: {
        totalRelevant: results.length,
        fallbackMode: true
      }
    };
  }
}

module.exports = new GraphRAG();
