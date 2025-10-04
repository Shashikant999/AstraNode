class GraphBuilder {
  buildGraph(papers, connections) {
    console.log('Building graph structure...');
    
    // Create nodes from papers
    const nodes = papers.map(paper => ({
      id: paper.id,
      title: paper.title,
      link: paper.link,
      concepts: paper.concepts || [],
      domain: paper.domain || 'Unknown',
      methodology: paper.methodology,
      degree: 0, // Will be calculated
      cluster: null // Will be assigned during clustering
    }));
    
    // Create edges with enhanced properties
    const links = connections.map(conn => ({
      source: conn.source,
      target: conn.target,
      strength: conn.strength,
      sharedConcepts: conn.sharedConcepts || [],
      type: this.getConnectionType(conn.strength)
    }));
    
    // Calculate node degrees
    const degreeMap = new Map();
    links.forEach(link => {
      degreeMap.set(link.source, (degreeMap.get(link.source) || 0) + 1);
      degreeMap.set(link.target, (degreeMap.get(link.target) || 0) + 1);
    });
    
    nodes.forEach(node => {
      node.degree = degreeMap.get(node.id) || 0;
    });
    
    // Perform clustering
    const clusteredNodes = this.performClustering(nodes, links);
    
    // Calculate centrality measures
    const centralityScores = this.calculateCentrality(clusteredNodes, links);
    
    clusteredNodes.forEach(node => {
      node.centrality = centralityScores.get(node.id) || 0;
    });
    
    return {
      nodes: clusteredNodes,
      links,
      stats: {
        totalNodes: clusteredNodes.length,
        totalLinks: links.length,
        averageDegree: nodes.reduce((sum, n) => sum + n.degree, 0) / nodes.length,
        clusters: [...new Set(clusteredNodes.map(n => n.cluster))].length
      }
    };
  }
  
  getConnectionType(strength) {
    if (strength > 0.7) return 'strong';
    if (strength > 0.5) return 'medium';
    return 'weak';
  }
  
  performClustering(nodes, links) {
    // Simple community detection using connected components and modularity
    const clusters = new Map();
    let clusterId = 0;
    
    // Build adjacency list
    const adjacencyList = new Map();
    nodes.forEach(node => adjacencyList.set(node.id, new Set()));
    
    links.forEach(link => {
      adjacencyList.get(link.source).add(link.target);
      adjacencyList.get(link.target).add(link.source);
    });
    
    // Find connected components
    const visited = new Set();
    
    nodes.forEach(node => {
      if (!visited.has(node.id)) {
        const component = this.dfs(node.id, adjacencyList, visited);
        
        // Assign cluster based on dominant domain in component
        const domainCounts = new Map();
        component.forEach(nodeId => {
          const nodeData = nodes.find(n => n.id === nodeId);
          const domain = nodeData.domain;
          domainCounts.set(domain, (domainCounts.get(domain) || 0) + 1);
        });
        
        const dominantDomain = [...domainCounts.entries()]
          .sort((a, b) => b[1] - a[1])[0][0];
        
        component.forEach(nodeId => {
          clusters.set(nodeId, `${dominantDomain}-${clusterId}`);
        });
        
        clusterId++;
      }
    });
    
    // Apply clusters to nodes
    return nodes.map(node => ({
      ...node,
      cluster: clusters.get(node.id) || 'unclustered'
    }));
  }
  
  dfs(nodeId, adjacencyList, visited, component = []) {
    visited.add(nodeId);
    component.push(nodeId);
    
    const neighbors = adjacencyList.get(nodeId) || new Set();
    neighbors.forEach(neighborId => {
      if (!visited.has(neighborId)) {
        this.dfs(neighborId, adjacencyList, visited, component);
      }
    });
    
    return component;
  }
  
  calculateCentrality(nodes, links) {
    // Calculate betweenness centrality (simplified)
    const centrality = new Map();
    const nodeIds = nodes.map(n => n.id);
    
    // Initialize
    nodeIds.forEach(id => centrality.set(id, 0));
    
    // Build adjacency list
    const adjacencyList = new Map();
    nodeIds.forEach(id => adjacencyList.set(id, new Set()));
    
    links.forEach(link => {
      adjacencyList.get(link.source).add(link.target);
      adjacencyList.get(link.target).add(link.source);
    });
    
    // Simple degree centrality (normalized)
    const maxDegree = Math.max(...nodes.map(n => n.degree));
    
    nodes.forEach(node => {
      const normalizedDegree = maxDegree > 0 ? node.degree / maxDegree : 0;
      centrality.set(node.id, normalizedDegree);
    });
    
    return centrality;
  }
  
  getRecommendations(graphData, targetNodeId, maxRecommendations = 10) {
    const targetNode = graphData.nodes.find(n => n.id === targetNodeId);
    if (!targetNode) return [];
    
    // Find direct connections
    const directConnections = graphData.links
      .filter(link => link.source === targetNodeId || link.target === targetNodeId)
      .map(link => ({
        nodeId: link.source === targetNodeId ? link.target : link.source,
        strength: link.strength,
        type: 'direct',
        sharedConcepts: link.sharedConcepts
      }));
    
    // Find indirect connections (2-hop)
    const indirectConnections = this.findIndirectConnections(graphData, targetNodeId, 2);
    
    // Combine and rank recommendations
    const allRecommendations = [...directConnections, ...indirectConnections];
    
    // Remove duplicates and sort by strength
    const uniqueRecommendations = new Map();
    
    allRecommendations.forEach(rec => {
      const existing = uniqueRecommendations.get(rec.nodeId);
      if (!existing || rec.strength > existing.strength) {
        uniqueRecommendations.set(rec.nodeId, rec);
      }
    });
    
    // Convert to array and add node data
    const recommendations = [...uniqueRecommendations.values()]
      .sort((a, b) => b.strength - a.strength)
      .slice(0, maxRecommendations)
      .map(rec => {
        const node = graphData.nodes.find(n => n.id === rec.nodeId);
        return {
          ...rec,
          node: node
        };
      });
    
    return recommendations;
  }
  
  findIndirectConnections(graphData, targetNodeId, maxHops) {
    const indirectConnections = [];
    const visited = new Set([targetNodeId]);
    
    // BFS to find indirect connections
    const queue = [{ nodeId: targetNodeId, path: [], strength: 1.0 }];
    
    while (queue.length > 0) {
      const current = queue.shift();
      
      if (current.path.length >= maxHops) continue;
      
      const currentConnections = graphData.links.filter(link => 
        link.source === current.nodeId || link.target === current.nodeId
      );
      
      currentConnections.forEach(link => {
        const nextNodeId = link.source === current.nodeId ? link.target : link.source;
        
        if (!visited.has(nextNodeId)) {
          const newPath = [...current.path, current.nodeId];
          const newStrength = current.strength * link.strength * 0.7; // Decay for indirect
          
          if (newPath.length === maxHops - 1) {
            // This is a valid indirect connection
            indirectConnections.push({
              nodeId: nextNodeId,
              strength: newStrength,
              type: `indirect-${newPath.length + 1}hop`,
              path: [...newPath, nextNodeId],
              sharedConcepts: link.sharedConcepts
            });
          } else {
            // Continue searching
            queue.push({
              nodeId: nextNodeId,
              path: newPath,
              strength: newStrength
            });
          }
          
          visited.add(nextNodeId);
        }
      });
    }
    
    return indirectConnections;
  }
  
  getClusterAnalysis(graphData) {
    const clusterStats = new Map();
    
    graphData.nodes.forEach(node => {
      const cluster = node.cluster;
      if (!clusterStats.has(cluster)) {
        clusterStats.set(cluster, {
          name: cluster,
          nodes: [],
          concepts: new Set(),
          avgCentrality: 0,
          totalDegree: 0
        });
      }
      
      const stats = clusterStats.get(cluster);
      stats.nodes.push(node);
      node.concepts.forEach(concept => stats.concepts.add(concept));
      stats.totalDegree += node.degree;
    });
    
    // Calculate averages
    clusterStats.forEach(stats => {
      stats.avgCentrality = stats.nodes.reduce((sum, n) => sum + (n.centrality || 0), 0) / stats.nodes.length;
      stats.concepts = [...stats.concepts];
      stats.size = stats.nodes.length;
    });
    
    return [...clusterStats.values()].sort((a, b) => b.size - a.size);
  }
}

module.exports = new GraphBuilder();
