🔑 **ALL API DETAILS & CONFIGURATION**
===============================================

## 📍 **1. ENVIRONMENT FILE (.env)**
**Location:** `/Users/shashikantnanda/sunhacks/langchain-agents/.env`

```properties
# 🤖 GOOGLE GEMINI API (Main AI Provider)
GEMINI_API_KEY=AIzaSyC6M93MsQLVXzx8JVQTlfN529udFufWp5w
GOOGLE_API_KEY=AIzaSyC6M93MsQLVXzx8JVQTlfN529udFufWp5w

# 🌐 EXISTING KNOWLEDGE GRAPH API  
GRAPHRAG_API_URL=http://localhost:3001

# ⚙️ AGENT CONFIGURATION
MAX_ITERATIONS=10
TEMPERATURE=0.7
```

## 📍 **2. GOOGLE CLOUD PROJECT**
**Project ID:** `gen-lang-client-0400019191`  
**Project Number:** `899103612067`

**Management Links:**
- **Billing:** https://console.cloud.google.com/billing/linkedaccount?project=gen-lang-client-0400019191
- **APIs:** https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project=gen-lang-client-0400019191
- **API Keys:** https://aistudio.google.com/app/apikey

## 📍 **3. API INTEGRATION POINTS**

### A. **Google Gemini Integration**
**Files:** `app/agents.py`, `app/agents_new.py`, `app/main.py`

**Key Code:**
```python
# API Key Loading (agents.py line 54)
self.google_api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# LangChain Integration
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=self.google_api_key,
    temperature=0.1
)
```

### B. **Knowledge Graph API**
**File:** `app/tools.py`

**Key Endpoints:**
```python
# GraphRAG API Interface (line 29-75)
class GraphRAGAPI:
    base_url = "http://localhost:3001"  # From GRAPHRAG_API_URL
    
    # Main endpoints:
    POST /api/rag/query          # Query the knowledge graph
    GET  /api/rag/concept/{id}   # Explore concepts  
    GET  /api/papers             # Get all papers
    GET  /api/search             # Search papers
```

### C. **FastAPI Server Endpoints**
**File:** `app/main.py`

**Your API Endpoints:**
```python
# Research System APIs
POST /langchain/query        # LangChain + Gemini queries
POST /gemini/query          # Direct Gemini queries  
GET  /health                # System health check
GET  /                      # Web interface
```

## 📍 **4. DEPENDENCIES (pyproject.toml)**
```toml
dependencies = [
    "langchain-google-genai>=2.0.0",  # Google Gemini integration
    "google-generativeai>=0.8.0",     # Direct Google AI SDK
    "fastapi>=0.100.0",               # API server
    "python-dotenv>=1.0.0",           # Environment variables
    "requests>=2.31.0",               # HTTP client for GraphRAG
    # ... data science libraries for analysis
]
```

## 📍 **5. CURRENT STATUS**

### ✅ **Working:**
- Environment configuration
- All dependencies installed
- LangChain integration code
- Knowledge graph connection (localhost:3001)
- Demo mode with 607 papers

### ⚠️ **Needs Setup:**
- Google Cloud billing enablement
- Generative Language API activation
- API key validation

## 📍 **6. QUICK REFERENCE**

### **Test API Status:**
```bash
uv run python check_status.py
```

### **Start Research System:**
```bash
uv run python -m app.main
# Opens at: http://localhost:8000
```

### **Enable Google APIs:**
1. **Billing:** https://console.cloud.google.com/billing/linkedaccount?project=gen-lang-client-0400019191
2. **API:** https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com?project=gen-lang-client-0400019191

## 📍 **7. API FLOW DIAGRAM**
```
User Query → FastAPI (port 8000) → LangChain Agent → Google Gemini API
                                                  ↓
Knowledge Graph (port 3001) ← Research Tools ←─┘
                     ↓
              Analysis + Response
```

**Summary:** All APIs are configured and ready. Just needs Google Cloud billing + API enablement to go live!
