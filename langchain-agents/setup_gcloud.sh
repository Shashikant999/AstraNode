#!/bin/bash
# 🚀 Google Cloud Setup Script for Project gen-lang-client-0400019191

echo "🔧 Setting up Google Cloud Project for Gemini API"
echo "Project: gen-lang-client-0400019191 (899103612067)"
echo "=================================================="

# Set the project
echo "📋 Setting active project..."
gcloud config set project gen-lang-client-0400019191

echo ""
echo "🚀 Enabling required APIs..."
gcloud services enable generativelanguage.googleapis.com
gcloud services enable aiplatform.googleapis.com

echo ""
echo "📊 Checking enabled services..."
gcloud services list --enabled --filter="name:generativelanguage OR name:aiplatform"

echo ""
echo "🔑 Checking current authentication..."
gcloud auth list

echo ""
echo "✅ Setup complete! Next steps:"
echo "1. Ensure billing is enabled: https://console.cloud.google.com/billing"
echo "2. Test API key: cd /Users/shashikantnanda/sunhacks/langchain-agents && uv run python test_enhanced.py"
echo "3. If still issues, regenerate API key at: https://aistudio.google.com/app/apikey"
