const OpenAI = require('openai');
const natural = require('natural');

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

class PaperAnalyzer {
  constructor() {
    this.tokenizer = natural.WordTokenizer;
    this.stemmer = natural.PorterStemmer;
  }

  async extractConcepts(papers) {
    console.log('Extracting concepts from papers...');
    
    // Process papers in batches to avoid API limits
    const batchSize = 5;
    const analyzedPapers = [];
    
    for (let i = 0; i < papers.length; i += batchSize) {
      const batch = papers.slice(i, i + batchSize);
      
      try {
        const batchResults = await this.processBatch(batch);
        analyzedPapers.push(...batchResults);
        
        console.log(`Processed ${analyzedPapers.length}/${papers.length} papers`);
        
        // Small delay to respect API rate limits
        if (i + batchSize < papers.length) {
          await this.delay(1000);
        }
      } catch (error) {
        console.error(`Error processing batch ${i}-${i + batchSize}:`, error);
        // Add papers without concepts if API fails
        analyzedPapers.push(...batch.map(paper => ({
          ...paper,
          concepts: this.extractBasicConcepts(paper.title)
        })));
      }
    }
    
    return analyzedPapers;
  }

  async processBatch(papers) {
    const titles = papers.map(p => p.title).join('\n');
    
    const prompt = `
    Analyze these research paper titles and extract key concepts, keywords, and research areas for each paper.
    For each title, identify:
    1. Main research domain (e.g., "microgravity", "space biology", "stem cells")
    2. Specific techniques or methods mentioned
    3. Biological systems or organisms studied
    4. Key processes or phenomena investigated
    
    Return a JSON array where each object contains:
    - title: the original title
    - concepts: array of 3-7 key concepts/keywords
    - domain: primary research domain
    - methodology: research approach if mentioned
    
    Paper titles:
    ${titles}
    `;

    try {
      const response = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.3,
        max_tokens: 2000
      });

      const content = response.choices[0].message.content.trim();
      
      // Try to parse JSON response
      let analysisResults;
      try {
        // Extract JSON from the response if it's wrapped in markdown
        const jsonMatch = content.match(/\[[\s\S]*\]/);
        const jsonStr = jsonMatch ? jsonMatch[0] : content;
        analysisResults = JSON.parse(jsonStr);
      } catch (parseError) {
        console.error('Failed to parse AI response:', parseError);
        throw new Error('Invalid AI response format');
      }

      // Merge with original paper data
      return papers.map((paper, index) => {
        const analysis = analysisResults[index] || {};
        return {
          ...paper,
          concepts: analysis.concepts || this.extractBasicConcepts(paper.title),
          domain: analysis.domain || 'Unknown',
          methodology: analysis.methodology || null
        };
      });
      
    } catch (error) {
      console.error('OpenAI API error:', error);
      throw error;
    }
  }

  extractBasicConcepts(title) {
    // Fallback method using NLP if AI fails
    const tokens = title.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 3)
      .filter(word => !this.isStopWord(word));
    
    const concepts = [...new Set(tokens)].slice(0, 5);
    return concepts;
  }

  async findConnections(papers) {
    console.log('Finding connections between papers...');
    
    const connections = [];
    
    // Calculate similarity between all pairs
    for (let i = 0; i < papers.length; i++) {
      for (let j = i + 1; j < papers.length; j++) {
        const similarity = this.calculateSimilarity(papers[i], papers[j]);
        
        if (similarity > 0.3) { // Threshold for connection
          connections.push({
            source: papers[i].id,
            target: papers[j].id,
            strength: similarity,
            sharedConcepts: this.getSharedConcepts(papers[i], papers[j])
          });
        }
      }
    }
    
    // Sort by strength and return top connections
    return connections
      .sort((a, b) => b.strength - a.strength)
      .slice(0, Math.min(500, connections.length)); // Limit for performance
  }

  calculateSimilarity(paper1, paper2) {
    const concepts1 = new Set(paper1.concepts.map(c => c.toLowerCase()));
    const concepts2 = new Set(paper2.concepts.map(c => c.toLowerCase()));
    
    const intersection = new Set([...concepts1].filter(x => concepts2.has(x)));
    const union = new Set([...concepts1, ...concepts2]);
    
    // Jaccard similarity
    const jaccardSim = intersection.size / union.size;
    
    // Domain similarity bonus
    const domainBonus = (paper1.domain === paper2.domain) ? 0.2 : 0;
    
    // Title similarity using cosine similarity
    const titleSim = this.cosineSimilarity(
      this.getTitleVector(paper1.title),
      this.getTitleVector(paper2.title)
    ) * 0.3;
    
    return jaccardSim + domainBonus + titleSim;
  }

  getTitleVector(title) {
    const words = title.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2 && !this.isStopWord(word));
    
    const vector = {};
    words.forEach(word => {
      vector[word] = (vector[word] || 0) + 1;
    });
    
    return vector;
  }

  cosineSimilarity(vec1, vec2) {
    const keys1 = Object.keys(vec1);
    const keys2 = Object.keys(vec2);
    const allKeys = [...new Set([...keys1, ...keys2])];
    
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    allKeys.forEach(key => {
      const val1 = vec1[key] || 0;
      const val2 = vec2[key] || 0;
      dotProduct += val1 * val2;
      norm1 += val1 * val1;
      norm2 += val2 * val2;
    });
    
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2)) || 0;
  }

  getSharedConcepts(paper1, paper2) {
    const concepts1 = new Set(paper1.concepts.map(c => c.toLowerCase()));
    const concepts2 = new Set(paper2.concepts.map(c => c.toLowerCase()));
    
    return [...concepts1].filter(c => concepts2.has(c));
  }

  isStopWord(word) {
    const stopWords = new Set([
      'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'
    ]);
    return stopWords.has(word.toLowerCase());
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = new PaperAnalyzer();
