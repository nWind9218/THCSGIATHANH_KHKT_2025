#!/bin/bash

echo ""
echo "========================================"
echo "   THCS GIA THANH - Railway Deployment"
echo "========================================"
echo ""

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] Git is not installed"
  exit 1
fi

if [ ! -f Dockerfile ]; then
  echo "[ERROR] Dockerfile not found"
  exit 1
fi

if [ ! -f requirements.txt ]; then
  echo "[ERROR] requirements.txt not found"
  exit 1
fi

echo "[OK] Required files look good"

echo ""
echo "[INFO] Git status:"
git status --short
echo ""

git add -A
if ! git diff-index --quiet --cached HEAD; then
  read -p "Enter commit message (default: 'Deploy to Railway'): " COMMIT_MSG
  COMMIT_MSG=${COMMIT_MSG:-"Deploy to Railway"}
  git commit -m "$COMMIT_MSG"
else
  echo "[INFO] No changes to commit"
fi

echo "[INFO] Pushing to GitHub..."
git push -u origin main
if [ $? -ne 0 ]; then
  echo "[ERROR] Git push failed"
  exit 1
fi

echo ""
echo "[OK] Push successful"
echo ""
echo "Next steps on Railway:"
echo "1. Open https://railway.com"
echo "2. Create New Project -> Deploy from GitHub repo"
echo "3. Ensure deployment uses Dockerfile"
echo "4. Set environment variables:"
echo "   - DATABASE_URL (Supabase Postgres)"
echo "   - REDIS_URL (Upstash Redis, rediss://)"
echo "   - OPENAI_API_KEY"
echo "   - ALLOWED_ORIGINS"
echo "5. Wait for service to become healthy"
echo ""
