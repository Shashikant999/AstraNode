#!/usr/bin/env python3
"""
NASA Space Apps GraphRAG - One-Click Setup
==========================================
This script automatically sets up your GraphRAG system with all dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    print(f"""
{Colors.BLUE}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════╗
║                NASA SPACE APPS GRAPHRAG SETUP                ║
║               Automated Installation Script                   ║
╚═══════════════════════════════════════════════════════════════╝
{Colors.END}
""")

def run_command(cmd, description, check=True):
    """Run a command with nice output"""
    print(f"{Colors.YELLOW}► {description}...{Colors.END}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{Colors.GREEN}✅ {description} completed successfully{Colors.END}")
            return True
        else:
            print(f"{Colors.RED}❌ {description} failed: {result.stderr}{Colors.END}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}❌ {description} failed: {e}{Colors.END}")
        return False

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"{Colors.GREEN}✅ Python {version.major}.{version.minor}.{version.micro} detected{Colors.END}")
        return True
    else:
        print(f"{Colors.RED}❌ Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}{Colors.END}")
        return False

def setup_virtual_environment():
    """Create and activate virtual environment"""
    if not os.path.exists(".venv"):
        if not run_command("python3 -m venv .venv", "Creating virtual environment"):
            return False
    else:
        print(f"{Colors.GREEN}✅ Virtual environment already exists{Colors.END}")
    
    # Install dependencies
    activate_cmd = "source .venv/bin/activate" if platform.system() != "Windows" else ".venv\\Scripts\\activate"
    pip_cmd = f"{activate_cmd} && pip install -r requirements.txt"
    
    return run_command(pip_cmd, "Installing Python dependencies")

def check_env_file():
    """Check for .env file and create from template if needed"""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if not env_path.exists():
        if env_example_path.exists():
            # Copy from .env.example
            import shutil
            shutil.copy(env_example_path, env_path)
            print(f"{Colors.YELLOW}⚠️  Created .env from template{Colors.END}")
        else:
            # Create basic template
            env_template = """# NASA Space Apps GraphRAG Configuration
# ===========================================

# Google Gemini API Key (Required)
# Get your free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Other API keys for enhanced functionality
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true
"""
            with open(env_path, 'w') as f:
                f.write(env_template)
            print(f"{Colors.YELLOW}⚠️  Created .env template file{Colors.END}")
        
        print(f"{Colors.RED}🔒 SECURITY: .env file is excluded from Git for safety{Colors.END}")
        print(f"{Colors.YELLOW}   Please add your API keys to .env file{Colors.END}")
        return False
    else:
        # Check if it contains real API key
        with open(env_path, 'r') as f:
            content = f.read()
        
        if "your_gemini_api_key_here" in content:
            print(f"{Colors.YELLOW}⚠️  .env exists but needs API key configuration{Colors.END}")
            return False
        else:
            print(f"{Colors.GREEN}✅ .env file configured{Colors.END}")
            return True

def create_run_script():
    """Create simple run script"""
    if platform.system() == "Windows":
        script_content = """@echo off
echo Starting NASA Space Apps GraphRAG...
cd /d "%~dp0"
call .venv\\Scripts\\activate
cd langchain-agents
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
"""
        script_name = "run.bat"
    else:
        script_content = """#!/bin/bash
echo "🚀 Starting NASA Space Apps GraphRAG..."
cd "$(dirname "$0")"
source .venv/bin/activate
cd langchain-agents
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
        script_name = "run.sh"
    
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    if platform.system() != "Windows":
        os.chmod(script_name, 0o755)
    
    print(f"{Colors.GREEN}✅ Created {script_name} launcher script{Colors.END}")

def main():
    print_header()
    
    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    success = True
    
    # Check Python
    if not check_python():
        success = False
    
    # Setup virtual environment
    if success and not setup_virtual_environment():
        success = False
    
    # Check environment file
    env_ready = check_env_file()
    
    # Create run script
    create_run_script()
    
    print(f"\n{Colors.BLUE}{Colors.BOLD}Setup Summary:{Colors.END}")
    
    if success and env_ready:
        print(f"{Colors.GREEN}🎉 Setup completed successfully!{Colors.END}")
        print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
        print(f"1. Run: {'./run.sh' if platform.system() != 'Windows' else 'run.bat'}")
        print(f"2. Open: http://localhost:8000")
        print(f"3. Start querying your NASA research database!")
        
    elif success and not env_ready:
        print(f"{Colors.YELLOW}⚠️  Setup almost complete!{Colors.END}")
        print(f"\n{Colors.BOLD}Required Action:{Colors.END}")
        print(f"1. Get a free Gemini API key: {Colors.BLUE}https://aistudio.google.com/app/apikey{Colors.END}")
        print(f"2. Add it to .env file: GEMINI_API_KEY=your_key_here")
        print(f"3. Run: {'./run.sh' if platform.system() != 'Windows' else 'run.bat'}")
        
    else:
        print(f"{Colors.RED}❌ Setup failed. Please check the errors above.{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()
