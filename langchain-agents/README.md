# Research Paper Knowledge Graph - LangChain Agents

Advanced research assistant using LangChain agents for intelligent research paper analysis.

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set environment variables:
```bash
# Copy from your GitHub Models API
export GITHUB_TOKEN="AlzaSyC6M93MsQLvXzx8JVQTIfN529udFufWp5w"
export OPENAI_API_KEY="AlzaSyC6M93MsQLvXzx8JVQTIfN529udFufWp5w"  # GitHub Models
```

3. Start the agent server:
```bash
uv run python app/server.py
```

## Features

- **Research Assistant Agent**: Multi-step reasoning about research papers
- **Citation Analysis Agent**: Trace paper influence and connections
- **Concept Explorer Agent**: Navigate research concepts intelligently
- **Gap Analysis Agent**: Identify research opportunities
- **Collaboration Agent**: Find potential research partnerships

## API Endpoints

- `/research_assistant` - General research questions
- `/citation_analysis` - Paper citation and influence analysis  
- `/concept_exploration` - Deep dive into research concepts
- `/gap_analysis` - Identify research gaps and opportunities
- `/collaboration_finder` - Find potential collaborators
