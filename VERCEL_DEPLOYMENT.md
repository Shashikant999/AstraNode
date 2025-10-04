# Vercel Deployment Guide

## Quick Deployment Steps

1. **Login to Vercel** (if not already logged in):
   ```bash
   vercel login
   ```
   Follow the prompts to login with your preferred method (GitHub, GitLab, Bitbucket, or Email).

2. **Deploy the Application**:
   ```bash
   vercel
   ```
   This will start the deployment process. You may be asked:
   - If you want to link to an existing project (choose No for first deployment)
   - Project name (you can use the default or enter "knovera-research-assistant")
   - Directory to deploy (press Enter for current directory)
   - Build settings (the vercel.json will handle this automatically)

3. **Production Deployment** (after initial deployment):
   ```bash
   vercel --prod
   ```

## What's Already Configured

✅ **vercel.json**: Updated to handle all your routes including:
- `/new` - Your new Knovera dashboard
- `/api/*` - All API endpoints
- `/gemini/*` - Gemini AI endpoints
- `/health` - Health check endpoint

✅ **Environment Variables**: Your Gemini API key is already configured

✅ **Dependencies**: requirements.txt is ready for Python dependencies

## Expected Deployment URL Structure

After deployment, your app will be available at:
- Main interface: `https://your-app-name.vercel.app/`
- Knovera Dashboard: `https://your-app-name.vercel.app/new`
- Health Check: `https://your-app-name.vercel.app/health`
- API endpoints: `https://your-app-name.vercel.app/api/...`

## Troubleshooting

If you encounter issues:
1. Check deployment logs: `vercel logs <deployment-url>`
2. Verify environment variables: Check Vercel dashboard
3. Test locally first: `vercel dev`

## Post-Deployment

Once deployed successfully, you can:
1. Access your Knovera Research Assistant at the provided URL
2. Share the `/new` endpoint for the enhanced dashboard
3. Monitor usage in Vercel dashboard
4. Set up custom domain if needed

Run `vercel` in this directory to start the deployment!
