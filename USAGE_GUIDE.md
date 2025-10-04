# 🧠 Research Paper Knowledge Graph - Usage Guide

Your intelligent research paper knowledge graph is now ready! Here's how to use it:

## 🚀 Getting Started

1. **Open the Application**: Visit http://localhost:3000 in your browser
2. **Load Your Data**: The system has automatically loaded 607 research papers from your CSV file
3. **Start Analysis**: Click the "Start AI Analysis" button to begin processing

## 🎯 Key Features

### 📊 AI-Powered Analysis
- **Concept Extraction**: Automatically identifies key research concepts from paper titles
- **Domain Classification**: Groups papers by research areas (e.g., Microgravity Research, Cell Biology)
- **Connection Detection**: Finds relationships between papers based on shared concepts and themes

### 🕸️ Interactive Knowledge Graph
- **Drag & Drop**: Click and drag nodes to explore the network
- **Zoom Controls**: Use the zoom buttons (top-left) to navigate the graph
- **Node Sizes**: Larger nodes indicate papers with more connections
- **Color Coding**: Different colors represent different research domains
- **Connection Strength**: Thicker lines indicate stronger relationships

### 🔍 Smart Features
- **Paper Details**: Click any node to see detailed information in the right panel
- **Related Papers**: Get AI-powered recommendations for similar research
- **Search & Filter**: Use the search panel (left drawer) to find specific papers or topics
- **Analytics**: View comprehensive graph statistics (right drawer)

## 🎮 How to Navigate

### Main Interface
1. **Graph Area**: The main visualization showing paper connections
2. **Paper Details Panel**: Opens when you click a node (right side)
3. **Search Panel**: Access via the search button in the toolbar
4. **Analytics Panel**: Access via the analytics button in the toolbar

### Graph Controls
- **Click**: Select a paper and view details
- **Drag**: Move papers around to reorganize the layout
- **Hover**: See quick information in tooltips
- **Zoom**: Use mouse wheel or zoom controls

### Search & Discovery
1. **Text Search**: Find papers by title or concept keywords
2. **Domain Filter**: Click domain chips to filter by research area
3. **Recommendations**: Click any paper to see related publications
4. **Connection Paths**: Trace how concepts connect across different papers

## 📈 Understanding the Analysis

### Connection Types
- 🟢 **Strong** (Green): High conceptual overlap (>70% similarity)
- 🟠 **Medium** (Orange): Moderate similarity (50-70%)
- ⚪ **Weak** (Gray): Basic connections (30-50%)

### Node Properties
- **Size**: Based on number of connections (degree centrality)
- **Color**: Research domain classification
- **Position**: Algorithmic layout based on relationships
- **Badges**: Red circles show highly connected "hub" papers

### Research Domains
The system automatically identifies domains such as:
- 🚀 **Microgravity Research**: Space-related studies
- 🦴 **Bone Research**: Skeletal and bone-related work
- 🧬 **Cell Biology**: Cellular and molecular studies
- 🌱 **Plant Biology**: Botanical research
- ⚕️ **Space Medicine**: Human health in space

## 💡 Tips for Best Results

1. **Start with Hubs**: Click on large, highly-connected nodes to discover central research themes
2. **Follow Connections**: Trace paths between different domains to find interdisciplinary opportunities
3. **Use Filters**: Narrow down to specific domains when exploring large datasets
4. **Check Analytics**: Review the analytics panel for insights about research trends
5. **Export Links**: Click "Open Paper" to access the original research publications

## 🔧 Troubleshooting

- **No Connections?**: Try lowering similarity thresholds in the code or check data quality
- **Slow Performance?**: Large datasets may take time to analyze and render
- **Missing Papers?**: Verify CSV format has "Title" and "Link" columns
- **Graph Layout**: Drag nodes to manually adjust the visualization

## 🎓 Advanced Usage

### For Researchers
- Identify research gaps by finding sparse connection areas
- Discover collaboration opportunities through shared concepts
- Track research evolution by analyzing connection patterns
- Find seminal papers through centrality analysis

### For Literature Reviews
- Automatically organize papers by theme
- Find related work through recommendation system
- Visualize research landscape and relationships
- Export structured data for systematic reviews

## 🔄 Current Mode: Demo Analysis
The system is currently running in **demo mode** with built-in NLP analysis. For enhanced AI capabilities:

1. Get an OpenAI API key from https://platform.openai.com/api-keys
2. Edit `.env` file and replace `demo_mode` with your API key
3. Restart the server for enhanced concept extraction and relationship detection

---

**🎉 Happy Exploring!** Your knowledge graph is ready to help you discover hidden connections in research literature.
