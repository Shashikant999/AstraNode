# 🚀 AstraNode Research Engine - Vercel Deployment Guide

## 🔧 Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install globally with `npm i -g vercel`
3. **Git Repository**: Ensure your code is in a Git repository

## 📋 Pre-Deployment Setup

### 1. Environment Variables
Set up the following environment variables in Vercel:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 2. Repository Requirements
- ✅ `vercel.json` - Deployment configuration
- ✅ `requirements.txt` - Python dependencies
- ✅ `runtime.txt` - Python version specification
- ✅ `.vercelignore` - Files to exclude from deployment

## 🚀 Deployment Steps

### Option 1: Deploy via Vercel CLI

```bash
# 1. Navigate to project directory
cd /path/to/AstraNode/Research_Engine

# 2. Login to Vercel
vercel login

# 3. Deploy (first time)
vercel

# 4. Deploy to production
vercel --prod
```

### Option 2: Deploy via Git Integration

1. **Push to GitHub/GitLab**:
   ```bash
   git add .
   git commit -m "🚀 Deploy AstraNode to Vercel"
   git push origin main
   ```

2. **Connect to Vercel**:
   - Go to [vercel.com/dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import your Git repository
   - Configure environment variables
   - Deploy!

## 🔐 Environment Variables Setup

In Vercel Dashboard:
1. Go to **Project Settings** → **Environment Variables**
2. Add the following variables:

| Name | Value | Environment |
|------|-------|-------------|
| `GEMINI_API_KEY` | `your_api_key_here` | Production, Preview, Development |
| `GOOGLE_API_KEY` | `your_api_key_here` | Production, Preview, Development |

## 🛠️ Configuration Details

### vercel.json Configuration
- **Runtime**: Python 3.9
- **Max Lambda Size**: 50MB (for ML models)
- **Max Duration**: 60 seconds (for AI processing)
- **Routes**: Configured for FastAPI routing

### Performance Optimizations
- Lightweight dependencies for serverless
- Piper TTS models excluded from deployment
- Fallback to browser TTS in production
- Optimized for cold start performance

## 🔍 Post-Deployment Verification

1. **Health Check**: Visit `https://your-domain.vercel.app/health`
2. **API Status**: Check `https://your-domain.vercel.app/api/health`
3. **Research Dashboard**: Test full functionality

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**:
   - Check Python version compatibility
   - Verify all dependencies in requirements.txt

2. **API Key Issues**:
   - Ensure environment variables are set correctly
   - Check API key permissions and quotas

3. **Cold Start Timeouts**:
   - Increase maxDuration in vercel.json
   - Optimize imports and initialization

4. **Large Bundle Size**:
   - Use .vercelignore to exclude unnecessary files
   - Consider splitting large dependencies

### Debug Commands

```bash
# Check deployment logs
vercel logs

# View project info
vercel ls

# Check environment variables
vercel env ls
```

## 📊 Expected Deployment Features

✅ **Research Dashboard** - Analytics and KPIs
✅ **Research Publications** - 607 space biology papers
✅ **Citation Analysis** - Interactive visualizations
✅ **Research Assistance** - AI-powered analysis
✅ **TTS Integration** - Summary podcast generation (fallback mode)

## 🌐 Production URL

After deployment, your AstraNode Research Engine will be available at:
`https://your-project-name.vercel.app`

## 📚 Additional Resources

- [Vercel Python Documentation](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/vercel/)
- [AstraNode GitHub Repository](https://github.com/International-NASA-Space-App-Hackathon/AstraNode)

---

**🎯 Ready for NASA Space Apps Hackathon! 🚀**
