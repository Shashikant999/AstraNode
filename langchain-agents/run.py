#!/usr/bin/env python3
"""
LangChain Research Agents Setup and Development Server

This script handles the setup and running of the LangChain research agents.
It will install dependencies if needed and start the FastAPI server.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_command(cmd, cwd=None, capture_output=False):
    """Run a shell command"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=True
        )
        if capture_output:
            return result.stdout.strip()
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        if capture_output:
            print(f"Error: {e.stderr}")
        return False


def check_dependencies():
    """Check if LangChain dependencies are installed"""
    try:
        import langchain
        import langchain_openai  
        import fastapi
        import uvicorn
        return True
    except ImportError:
        return False


def install_dependencies():
    """Install dependencies using uv"""
    print("Installing LangChain dependencies...")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("Error: pyproject.toml not found. Make sure you're in the langchain-agents directory.")
        return False
    
    # Install dependencies
    if not run_command("uv sync"):
        print("Failed to install dependencies with uv sync")
        return False
    
    print("Dependencies installed successfully!")
    return True


def setup_environment():
    """Setup environment variables"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("Warning: .env file not found. Please create one with your GitHub Models API key.")
        return False
    
    # Load environment variables
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value.strip('"\'')
    
    return True


def start_server():
    """Start the FastAPI server"""
    print("Starting LangChain Research Agents server...")
    
    # Set up environment
    if not setup_environment():
        print("Warning: Environment setup incomplete")
    
    # Start with uv run to ensure proper environment
    cmd = "uv run python -m app.main"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Server failed to start: {e}")
        return False
    
    return True


def main():
    """Main setup and run function"""
    print("🚀 LangChain Research Agents Setup")
    print("=" * 40)
    
    # Check current directory
    if not Path("pyproject.toml").exists():
        print("Error: Not in langchain-agents directory")
        print("Please run this from the langchain-agents folder")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("📦 Dependencies not found, installing...")
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            sys.exit(1)
    else:
        print("✅ Dependencies already installed")
    
    # Start server
    print("\n🌐 Starting server...")
    if not start_server():
        print("❌ Failed to start server")
        sys.exit(1)


def show_help():
    """Show help information"""
    print("""
LangChain Research Agents - Setup and Run

Commands:
  python run.py              - Install deps (if needed) and start server
  python run.py --install    - Install dependencies only
  python run.py --server     - Start server only (skip dependency check)
  python run.py --help       - Show this help

Environment Setup:
  1. Make sure you have a .env file with GITHUB_MODELS_API_KEY
  2. Run 'python run.py' to install deps and start server
  3. Server will start on http://localhost:8000

API Endpoints:
  GET  /                     - Health check
  POST /agent/query          - Query any agent
  POST /agent/research       - Research assistant
  POST /agent/explore-concept - Concept exploration  
  POST /agent/find-collaborations - Find collaborations
  POST /agent/analyze        - Deep analysis
  GET  /tools               - List available tools
  GET  /agents              - List agent status

Integration with Knowledge Graph:
  The agents connect to your existing GraphRAG API at localhost:3001
  Make sure your main knowledge graph server is running first!
    """)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--help":
            show_help()
        elif arg == "--install":
            print("Installing dependencies only...")
            if install_dependencies():
                print("✅ Installation complete")
            else:
                print("❌ Installation failed")
                sys.exit(1)
        elif arg == "--server":
            print("Starting server only...")
            start_server()
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
            sys.exit(1)
    else:
        main()
