@echo off
REM NASA Space Apps GraphRAG - Windows Quick Start
REM ===============================================

echo 🚀 Starting NASA Space Apps GraphRAG System...
echo.

REM Get script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found!
    echo    Please run: python setup.py
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo ❌ .env file not found!
    echo    Please run: python setup.py
    pause
    exit /b 1
)

REM Check for API key (basic check)
findstr /C:"GEMINI_API_KEY=" .env | findstr /V /C:"your_gemini_api_key_here" > nul
if %errorlevel% neq 0 (
    echo ⚠️  Please add your Gemini API key to .env file
    echo    Get your free key at: https://aistudio.google.com/app/apikey
    echo    Then edit .env and set: GEMINI_API_KEY=your_actual_key
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating Python environment...
call .venv\Scripts\activate

REM Navigate to app directory
cd langchain-agents

REM Start the server
echo 🌐 Starting FastAPI server on http://localhost:8000...
echo.
echo ✅ Your GraphRAG system is ready!
echo    📊 607 NASA research papers loaded
echo    🤖 AI-powered analysis with Google Gemini
echo    📈 Interactive knowledge graphs
echo.
echo 🌍 Open in browser: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
