# 🚀 NASA Space Apps Challenge 2025: Space Biology GraphRAG

**Intelligent Research Assistant for NASA Space Biology Research**  
**Challenge Submission**: October 4, 2025  

An AI-powered GraphRAG system that analyzes 607 real NASA space biology research papers using Google Gemini 2.5 Pro, enabling intelligent knowledge discovery for current ISS research and future Mars missions.

## 🎯 NASA Challenge Features

- 🧠 **Gemini 2.5 Pro AI**: Advanced reasoning with thinking capabilities for space biology analysis
- � **607 NASA Papers**: Real space biology research from ISS, Bion missions, and NASA GeneLab
- � **Space Research Focus**: Microgravity, radiation effects, bone loss, cardiovascular changes
- 🕸️ **Knowledge Graph**: Interactive visualization of space biology research connections  
- � **Research Discovery**: Multi-hop reasoning across space life sciences literature
- 📊 **Mission Planning**: Evidence-based insights for Mars missions and long-term habitation
- 🛤️ **Research Pathways**: Trace scientific discoveries and identify research gaps
- 🤖 **Natural Language**: Ask questions about space biology and get AI-powered answers
- 🎯 **Semantic Search**: Find papers by biological concepts, not just keywords
- 🌐 **Cross-Domain**: Connect molecular, cellular, and systems-level space research
- 💡 **Future Missions**: Generate insights for countermeasure development and risk assessment

## Setup

1. Install dependencies:
```bash
npm install
npm run install-client
```

2. Create `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
PORT=5000
```

3. Run the application:
```bash
# Option 1: Use the startup script
./start-app.sh

# Option 2: Start servers separately
# Terminal 1 (Backend):
npm start

# Terminal 2 (Frontend):
cd client && npm start
```

## Usage

1. Upload your research papers CSV file (Title, Link columns)
2. Let the AI analyze and create connections between papers
3. Explore the interactive knowledge graph
4. Click on nodes to see paper details and related publications
5. Use the search and filtering features to find specific topics

## Technology Stack

- **Backend**: Node.js, Express, OpenAI API
- **Frontend**: React, D3.js, Material-UI
- **AI**: OpenAI GPT for concept extraction and similarity analysis
- **Visualization**: D3.js force-directed graph
