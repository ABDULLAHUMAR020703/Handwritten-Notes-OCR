# Deployment Guide

## Memory Requirements

**⚠️ CRITICAL**: This application requires **at least 2GB RAM** to run properly due to ML dependencies:
- PyTorch (~500MB-1GB)
- PaddleOCR (~500MB-1GB)
- Transformers models (~200-500MB)
- Other dependencies (~200MB)

**Render Free Tier (512MB) is insufficient** and will cause "Out of memory" errors.

## Deployment Options

### Option 1: Render (Paid Plan Required)
1. **Upgrade to Render Starter Plan** ($7/month) - provides 512MB-2GB RAM
2. Or use **Render Standard Plan** ($25/month) - provides 2GB-8GB RAM (recommended)

**Steps:**
1. Go to Render Dashboard → Your Service → Settings
2. Change instance type to **Starter** or **Standard**
3. Redeploy

### Option 2: Railway (Recommended for Free Tier)
Railway offers **512MB free** but you can upgrade to **$5/month for 2GB RAM** (cheaper than Render).

**Steps:**
1. Sign up at https://railway.app
2. Connect GitHub repo
3. Set root directory: `backend`
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables (especially `ALIBABA_API_KEY`)

### Option 3: Fly.io (Free Tier with More RAM)
Fly.io offers **256MB free** but allows you to scale up easily.

**Steps:**
1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Run `fly launch` in the `backend` directory
3. Scale memory: `fly scale memory 2048` (2GB)

### Option 4: Docker + Any Platform
Create a Dockerfile to have more control over the environment.

## Environment Variables

Set these in your hosting platform:

- `ALIBABA_API_KEY` - **REQUIRED** for OCR functionality
- `GEMINI_API_KEY` - Optional (for text cleanup)
- `PYTHON_VERSION` - Set to `3.10.13` (if platform supports it)

## Frontend Deployment (Vercel)

1. Deploy frontend to Vercel (free tier works fine)
2. Set environment variable:
   - `NEXT_PUBLIC_API_URL` = `https://your-backend-url.com`
3. Deploy

## Troubleshooting

### "Out of memory" error
- **Solution**: Upgrade to a plan with at least 2GB RAM

### "No open ports detected"
- **Solution**: Ensure start command uses `--port $PORT` (not hardcoded 8000)
- Check that the app responds to health check at `/` endpoint

### Build fails with Python version errors
- **Solution**: Ensure `.python-version` file exists with `3.10.13`
- Or set `PYTHON_VERSION=3.10.13` environment variable
