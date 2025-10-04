#!/bin/bash

echo "🚀 Setting up Research Paper Knowledge Graph..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js found: $(node --version)"

# Install backend dependencies
echo "📦 Installing backend dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install backend dependencies"
    exit 1
fi

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd client && npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install frontend dependencies"
    cd ..
    exit 1
fi

cd ..

# Check if CSV file exists
if [ ! -f "SB_publication_PMC.csv" ]; then
    echo "⚠️  CSV file 'SB_publication_PMC.csv' not found in project root"
    echo "   Please make sure your CSV file is in the correct location"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found"
    echo "   Creating .env template..."
    cat > .env << EOL
OPENAI_API_KEY=your_openai_api_key_here
PORT=5000
EOL
    echo "   Please edit .env file and add your OpenAI API key"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Make sure SB_publication_PMC.csv is in the project root"
echo "3. Run 'npm run dev' to start the application"
echo ""
echo "📚 The application will be available at http://localhost:3000"
echo "🔧 Backend API will be available at http://localhost:5000"
