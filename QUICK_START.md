# 🚀 NASA Space Apps GraphRAG - Plug & Play Setup

**One-click setup for your AI-powered space research system!**

## 📋 Quick Start (3 Steps)

### Step 1: Get API Key (Free)
1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key" 
4. Copy your key

### Step 2: Run Setup
```bash
python3 setup.py
```

### Step 3: Add API Key & Run
1. Edit `.env` file and add your key:
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```
2. Start the system:
   ```bash
   ./run.sh        # Mac/Linux
   run.bat         # Windows
   ```

🎉 **That's it!** Open http://localhost:8000

---

## 🎯 What You Get

- **607 Real NASA Papers** - Space biology research database
- **AI Analysis** - Powered by Google Gemini 2.5 Flash  
- **Interactive Graphs** - Visual knowledge connections
- **Natural Language Queries** - Ask questions in plain English
- **Clean Interface** - Optimized for research

---

## 💡 Example Queries

- "What are the effects of microgravity on plant growth?"
- "How does space radiation affect human cells?"
- "Show me research on Mars soil experiments"
- "Find papers about space agriculture"

---

## 🔧 Manual Setup (Alternative)

If automatic setup fails:

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start server
cd langchain-agents
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📂 Project Structure

```
NASA_Space_Apps_2025_GraphRAG/
├── setup.py                    # 🔧 Automated setup script
├── run.sh / run.bat            # 🚀 One-click launcher
├── .env                        # 🔑 API keys (auto-created)
├── requirements.txt            # 📦 Dependencies
├── SB_publication_PMC.csv      # 📊 NASA research database
└── langchain-agents/app/       # 🧠 Core application
    ├── main.py                 # FastAPI server
    ├── gemini_agent.py         # AI integration
    └── paper_database.py       # Research engine
```

---

## ⚡ System Features

- **Zero Configuration** - Automated setup handles everything
- **Cross-Platform** - Works on Mac, Linux, Windows
- **Hot Reload** - Auto-refresh during development  
- **Error Handling** - Clear error messages and solutions
- **Production Ready** - Optimized and secure

---

## 🆘 Troubleshooting

**Server won't start?**
```bash
# Check if port 8000 is free
lsof -i :8000           # Mac/Linux
netstat -ano | find ":8000"  # Windows

# Kill conflicting process
pkill -f uvicorn        # Mac/Linux
```

**API key issues?**
- Ensure no spaces around the `=` in `.env`
- Check key is valid at https://aistudio.google.com/
- Restart server after changing `.env`

**Dependencies missing?**
```bash
source .venv/bin/activate
pip install -r requirements.txt --upgrade
```

---

## 🌟 Share Your System

Your GraphRAG system is now plug-and-play! Anyone can:

1. **Clone your repo**
2. **Run `python3 setup.py`**  
3. **Add their API key**
4. **Launch with `./run.sh`**

Perfect for demos, hackathons, and research collaboration! 🎯
