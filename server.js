const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
require('dotenv').config();

const paperAnalyzer = require('./utils/paperAnalyzer');
const demoAnalyzer = require('./utils/demoAnalyzer');
const graphBuilder = require('./utils/graphBuilder');
const graphRAG = require('./utils/graphRAG');

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());
app.use(express.static('client/build'));

// Store papers and graph data in memory (for demo purposes)
let papers = [];
let graphData = null;

// Load initial CSV data
const loadPapers = () => {
  const csvPath = path.join(__dirname, 'SB_publication_PMC.csv');
  return new Promise((resolve, reject) => {
    const results = [];
    
    if (!fs.existsSync(csvPath)) {
      reject(new Error('CSV file not found'));
      return;
    }

    fs.createReadStream(csvPath)
      .pipe(csv({ skipEmptyLines: true }))
      .on('data', (data) => {
        // Handle BOM and different field names
        const titleKey = data['Title'] || data['﻿Title'] || data.title;
        const linkKey = data['Link'] || data.link;
        
        if (titleKey && linkKey) {
          results.push({
            id: results.length + 1,
            title: titleKey.trim(),
            link: linkKey.trim(),
            concepts: [],
            connections: []
          });
        }
      })
      .on('end', () => {
        console.log(`Loaded ${results.length} papers`);
        resolve(results);
      })
      .on('error', reject);
  });
};

// API Routes
app.get('/api/papers', (req, res) => {
  res.json(papers);
});

app.get('/api/graph', (req, res) => {
  res.json(graphData);
});

app.post('/api/analyze', async (req, res) => {
  try {
    console.log('Starting analysis...');
    
    // Choose analyzer based on API key availability
    const analyzer = process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY !== 'your_openai_api_key_here' 
      ? paperAnalyzer 
      : demoAnalyzer;
    
    console.log(`Using ${analyzer === paperAnalyzer ? 'OpenAI' : 'Demo'} analyzer`);
    
    // Extract concepts from paper titles
    const analyzedPapers = await analyzer.extractConcepts(papers);
    
    // Build connections between papers
    const connections = await analyzer.findConnections(analyzedPapers);
    
    // Create graph data structure
    graphData = graphBuilder.buildGraph(analyzedPapers, connections);
    
    // Initialize GraphRAG with the new data
    graphRAG.setGraphData(graphData, analyzedPapers);
    
    console.log(`Analysis complete. Found ${connections.length} connections.`);
    
    res.json({
      success: true,
      papersCount: analyzedPapers.length,
      connectionsCount: connections.length,
      graphData,
      analyzerType: analyzer === paperAnalyzer ? 'openai' : 'demo'
    });
  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

app.get('/api/recommendations/:paperId', (req, res) => {
  const paperId = parseInt(req.params.paperId);
  
  if (!graphData) {
    return res.status(400).json({ error: 'Graph not analyzed yet' });
  }
  
  const recommendations = graphBuilder.getRecommendations(graphData, paperId);
  res.json(recommendations);
});

app.get('/api/search', (req, res) => {
  const { query } = req.query;
  
  if (!query || !papers.length) {
    return res.json([]);
  }
  
  const results = papers.filter(paper => 
    paper.title.toLowerCase().includes(query.toLowerCase()) ||
    (paper.concepts && paper.concepts.some(concept => 
      concept.toLowerCase().includes(query.toLowerCase())
    ))
  );
  
  res.json(results);
});

// GraphRAG endpoints
app.post('/api/rag/query', async (req, res) => {
  try {
    const { query, options = {} } = req.body;
    
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }
    
    if (!graphData) {
      return res.status(400).json({ error: 'Graph not analyzed yet' });
    }
    
    const results = await graphRAG.queryGraph(query, options);
    res.json(results);
    
  } catch (error) {
    console.error('GraphRAG query error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/rag/concept/:concept', async (req, res) => {
  try {
    const { concept } = req.params;
    const { depth = 2 } = req.query;
    
    if (!graphData) {
      return res.status(400).json({ error: 'Graph not analyzed yet' });
    }
    
    const results = await graphRAG.exploreConceptNeighborhood(concept, parseInt(depth));
    res.json(results);
    
  } catch (error) {
    console.error('Concept exploration error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/rag/paths/:sourceId/:targetId', async (req, res) => {
  try {
    const { sourceId, targetId } = req.params;
    const { maxHops = 3 } = req.query;
    
    if (!graphData) {
      return res.status(400).json({ error: 'Graph not analyzed yet' });
    }
    
    const paths = await graphRAG.findResearchPaths(
      parseInt(sourceId), 
      parseInt(targetId), 
      parseInt(maxHops)
    );
    
    res.json(paths);
    
  } catch (error) {
    console.error('Path finding error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Serve React app (only in production)
app.get('*', (req, res) => {
  const buildPath = path.join(__dirname, 'client/build', 'index.html');
  if (fs.existsSync(buildPath)) {
    res.sendFile(buildPath);
  } else {
    res.json({ 
      message: 'API Server Running', 
      note: 'React app should be served separately in development mode' 
    });
  }
});

// Initialize server
const startServer = async () => {
  try {
    papers = await loadPapers();
    
    app.listen(PORT, () => {
      console.log(`🚀 Server running on port ${PORT}`);
      console.log(`📊 Loaded ${papers.length} research papers`);
      console.log(`🔗 Visit http://localhost:${PORT} to start analyzing`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
};

startServer();
