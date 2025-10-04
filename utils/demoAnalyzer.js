const natural = require('natural');

class DemoAnalyzer {
  constructor() {
    this.tokenizer = natural.WordTokenizer;
    this.stemmer = natural.PorterStemmer;
    
    // Demo keywords for different domains
    this.domainKeywords = {
      'Microgravity Research': ['microgravity', 'space', 'weightless', 'zero gravity', 'orbital'],
      'Bone Research': ['bone', 'skeletal', 'osteoblast', 'osteoclast', 'calcium', 'vertebrae'],
      'Cell Biology': ['cell', 'cellular', 'stem cell', 'differentiation', 'regeneration'],
      'Molecular Biology': ['gene', 'protein', 'RNA', 'DNA', 'expression', 'molecular'],
      'Plant Biology': ['arabidopsis', 'plant', 'root', 'pollen', 'golgi', 'vacuolar'],
      'Space Medicine': ['spaceflight', 'astronaut', 'radiation', 'mission', 'ISS'],
      'Developmental Biology': ['embryonic', 'development', 'growth', 'morphology'],
      'Genetics': ['genetic', 'genome', 'mutation', 'allele', 'chromosome']
    };
  }

  async extractConcepts(papers) {
    console.log('Using demo analyzer (no OpenAI API required)...');
    
    return papers.map(paper => {
      const concepts = this.extractBasicConcepts(paper.title);
      const domain = this.determineDomain(paper.title, concepts);
      
      return {
        ...paper,
        concepts: concepts,
        domain: domain,
        methodology: this.guessMethodology(paper.title)
      };
    });
  }

  extractBasicConcepts(title) {
    // Tokenize and clean
    const tokens = title.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2)
      .filter(word => !this.isStopWord(word));
    
    // Extract important terms
    const concepts = [];
    
    // Look for scientific terms
    tokens.forEach(token => {
      if (this.isScientificTerm(token)) {
        concepts.push(token);
      }
    });
    
    // Extract compound terms (bigrams)
    for (let i = 0; i < tokens.length - 1; i++) {
      const bigram = `${tokens[i]} ${tokens[i + 1]}`;
      if (this.isImportantBigram(bigram)) {
        concepts.push(bigram);
      }
    }
    
    // Remove duplicates and limit
    return [...new Set(concepts)].slice(0, 6);
  }

  determineDomain(title, concepts) {
    const titleLower = title.toLowerCase();
    let bestDomain = 'General Research';
    let maxScore = 0;
    
    Object.entries(this.domainKeywords).forEach(([domain, keywords]) => {
      let score = 0;
      
      keywords.forEach(keyword => {
        if (titleLower.includes(keyword.toLowerCase())) {
          score += keyword.length; // Longer, more specific terms get higher scores
        }
      });
      
      // Bonus for concept matches
      concepts.forEach(concept => {
        keywords.forEach(keyword => {
          if (concept.toLowerCase().includes(keyword.toLowerCase()) || 
              keyword.toLowerCase().includes(concept.toLowerCase())) {
            score += 2;
          }
        });
      });
      
      if (score > maxScore) {
        maxScore = score;
        bestDomain = domain;
      }
    });
    
    return bestDomain;
  }

  guessMethodology(title) {
    const titleLower = title.toLowerCase();
    
    if (titleLower.includes('validation') || titleLower.includes('method')) {
      return 'Experimental validation';
    } else if (titleLower.includes('analysis') || titleLower.includes('study')) {
      return 'Analytical study';
    } else if (titleLower.includes('effect') || titleLower.includes('response')) {
      return 'Response analysis';
    } else if (titleLower.includes('isolation') || titleLower.includes('purification')) {
      return 'Isolation/purification';
    }
    
    return null;
  }

  isScientificTerm(word) {
    // List of common scientific prefixes/suffixes
    const scientificPatterns = [
      /^micro/, /^nano/, /^bio/, /^neuro/, /^cardio/, /^hepato/,
      /osis$/, /itis$/, /emia$/, /tion$/, /ase$/, /oma$/,
      /gene/, /protein/, /enzyme/, /hormone/, /receptor/,
      /cell/, /tissue/, /organ/, /system/, /pathway/
    ];
    
    return scientificPatterns.some(pattern => pattern.test(word)) || 
           word.length > 8; // Longer words are often scientific terms
  }

  isImportantBigram(bigram) {
    const importantBigrams = [
      'stem cell', 'bone loss', 'gene expression', 'cell cycle', 
      'space flight', 'micro gravity', 'oxidative stress', 
      'real time', 'space station', 'tissue regeneration'
    ];
    
    return importantBigrams.includes(bigram.toLowerCase());
  }

  async findConnections(papers) {
    console.log('Finding connections using demo analyzer...');
    
    const connections = [];
    
    for (let i = 0; i < papers.length; i++) {
      for (let j = i + 1; j < papers.length; j++) {
        const similarity = this.calculateSimilarity(papers[i], papers[j]);
        
        if (similarity > 0.2) { // Lower threshold for demo
          connections.push({
            source: papers[i].id,
            target: papers[j].id,
            strength: similarity,
            sharedConcepts: this.getSharedConcepts(papers[i], papers[j])
          });
        }
      }
    }
    
    return connections
      .sort((a, b) => b.strength - a.strength)
      .slice(0, Math.min(300, connections.length));
  }

  calculateSimilarity(paper1, paper2) {
    const concepts1 = new Set(paper1.concepts.map(c => c.toLowerCase()));
    const concepts2 = new Set(paper2.concepts.map(c => c.toLowerCase()));
    
    const intersection = new Set([...concepts1].filter(x => concepts2.has(x)));
    const union = new Set([...concepts1, ...concepts2]);
    
    // Jaccard similarity
    const jaccardSim = intersection.size / union.size;
    
    // Domain similarity bonus
    const domainBonus = (paper1.domain === paper2.domain) ? 0.3 : 0;
    
    // Title word overlap
    const title1Words = new Set(paper1.title.toLowerCase().split(/\s+/));
    const title2Words = new Set(paper2.title.toLowerCase().split(/\s+/));
    const titleIntersection = new Set([...title1Words].filter(x => title2Words.has(x)));
    const titleSim = titleIntersection.size / Math.max(title1Words.size, title2Words.size) * 0.2;
    
    return Math.min(1.0, jaccardSim + domainBonus + titleSim);
  }

  getSharedConcepts(paper1, paper2) {
    const concepts1 = new Set(paper1.concepts.map(c => c.toLowerCase()));
    const concepts2 = new Set(paper2.concepts.map(c => c.toLowerCase()));
    
    return [...concepts1].filter(c => concepts2.has(c));
  }

  isStopWord(word) {
    const stopWords = new Set([
      'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
      'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
      'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 
      'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'may', 'have',
      'from', 'they', 'been', 'than', 'into', 'also', 'made', 'only', 'over',
      'time', 'very', 'what', 'with', 'will', 'would', 'there', 'could'
    ]);
    return stopWords.has(word.toLowerCase());
  }
}

module.exports = new DemoAnalyzer();
