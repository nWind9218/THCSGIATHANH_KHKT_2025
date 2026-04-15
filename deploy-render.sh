#!/bin/bash
# Deploy to Render - Automated Setup Script (macOS/Linux)

echo ""
echo "========================================"
echo "   THCS GIA THANH - Render Deployment"
echo "========================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "[ERROR] Git is not installed"
    echo "Install from: https://git-scm.com/download/linux"
    exit 1
fi

echo "[OK] Git is installed: $(git --version)"

# Check if .git exists
if [ ! -d .git ]; then
    echo "[INFO] Initializing Git repository..."
    git init
fi

# Check remote
if ! git remote -v | grep -q "origin"; then
    echo ""
    echo "[WARNING] Git remote 'origin' not set!"
    echo ""
    echo "You need to add your GitHub repository:"
    echo "  git remote add origin https://github.com/YOUR_USERNAME/THCS_GIATHANH_API.git"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "[OK] Git remote is configured"

# Check if required files exist
if [ ! -f requirements.txt ]; then
    echo "[ERROR] requirements.txt not found!"
    echo "Make sure you are in: THCSGIATHANH_KHKT_2025 directory"
    exit 1
fi

if [ ! -f Dockerfile ]; then
    echo "[ERROR] Dockerfile not found!"
    exit 1
fi

if [ ! -f render.yaml ]; then
    echo "[ERROR] render.yaml not found!"
    exit 1
fi

echo "[OK] All required files exist"

# Check git status
echo ""
echo "[INFO] Git status:"
git status --short
echo ""

# Prepare for deployment
echo "[INFO] Checking for uncommitted changes..."

# Try to add all files
git add -A

# Check if there are changes to commit
if ! git diff-index --quiet --cached HEAD; then
    echo "[INFO] Found uncommitted changes. Committing..."
    
    read -p "Enter commit message (default: 'Deploy to Render'): " COMMIT_MSG
    COMMIT_MSG=${COMMIT_MSG:-"Deploy to Render"}
    
    git commit -m "$COMMIT_MSG"
else
    echo "[INFO] No changes to commit"
fi

# Push to GitHub
echo ""
echo "[INFO] Pushing to GitHub..."
git push -u origin main

if [ $? -ne 0 ]; then
    echo "[ERROR] Git push failed"
    echo "Make sure your GitHub credentials are configured"
    exit 1
fi

echo "[OK] Push successful"

echo ""
echo "========================================"
echo "    Deployment Initiated"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Go to https://dashboard.render.com"
echo "2. Open your Web Service and set Environment Variables"
echo "3. Use Supabase Postgres URL for DATABASE_URL"
echo "4. Use Upstash Redis URL for REDIS_URL"
echo "5. Check logs and wait for build to complete (5-10 minutes)"
echo "6. Test the health endpoint:"
echo "   curl https://thcs-giathanh-api.onrender.com/health"
echo ""
echo "Environment Variables to Set:"
echo "  - DATABASE_URL (Supabase Postgres)"
echo "  - REDIS_URL (Upstash Redis)"
echo "  - OPENAI_API_KEY"
echo "  - SUPABASE_DB_URL (optional)"
echo "  - SUPABASE_API_KEY (optional)"
echo "  - ALLOWED_ORIGINS (Frontend URL)"
echo ""
