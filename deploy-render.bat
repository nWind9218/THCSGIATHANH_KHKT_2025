@REM Deploy to Render - Automated Setup Script (Windows)
@REM Run this after setting up Render services

@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================
echo    THCS GIA THANH - Render Deployment
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in PATH
    echo Install from: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [OK] Git is installed

REM Check if .git exists
if not exist .git (
    echo [INFO] Initializing Git repository...
    git init
)

REM Check remote
git remote -v | find "origin" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Git remote 'origin' not set!
    echo.
    echo You need to add your GitHub repository:
    echo   git remote add origin https://github.com/YOUR_USERNAME/THCS_GIATHANH_API.git
    echo.
    echo Then run this script again.
    pause
    exit /b 1
)

echo [OK] Git remote is configured

REM Check if requirements.txt exists
if not exist requirements.txt (
    echo [ERROR] requirements.txt not found!
    echo Make sure you are in: THCSGIATHANH_KHKT_2025 directory
    pause
    exit /b 1
)

REM Check if Dockerfile exists
if not exist Dockerfile (
    echo [ERROR] Dockerfile not found!
    pause
    exit /b 1
)

REM Check if render.yaml exists
if not exist render.yaml (
    echo [ERROR] render.yaml not found!
    pause
    exit /b 1
)

echo [OK] All required files exist

REM Check git status
echo.
echo [INFO] Git status:
git status --short
echo.

REM Prepare for deployment
echo [INFO] Checking for uncommitted changes...

REM Try to add all files
git add -A

REM Check if there are changes to commit
git diff-index --quiet --cached HEAD
if %errorlevel% neq 0 (
    echo [INFO] Found uncommitted changes. Committing...
    
    set /p COMMIT_MSG="Enter commit message (default: 'Deploy to Render'): "
    if "!COMMIT_MSG!"=="" set COMMIT_MSG=Deploy to Render
    
    git commit -m "!COMMIT_MSG!"
) else (
    echo [INFO] No changes to commit
)

REM Push to GitHub
echo.
echo [INFO] Pushing to GitHub...
git push -u origin main

if %errorlevel% neq 0 (
    echo [ERROR] Git push failed
    echo Make sure your GitHub credentials are configured
    pause
    exit /b 1
)

echo [OK] Push successful

echo.
echo ========================================
echo    Deployment Initiated
echo ========================================
echo.
echo Next steps:
echo 1. Go to https://dashboard.render.com
echo 2. Open your Web Service and set Environment Variables
echo 3. Use Supabase Postgres URL for DATABASE_URL
echo 4. Use Upstash Redis URL for REDIS_URL
echo 5. Check logs and wait for build to complete (5-10 minutes)
echo 6. Test the health endpoint:
echo    curl https://thcs-giathanh-api.onrender.com/health
echo.
echo Environment Variables to Set:
echo   - DATABASE_URL (Supabase Postgres)
echo   - REDIS_URL (Upstash Redis)
echo   - OPENAI_API_KEY
echo   - SUPABASE_DB_URL (optional)
echo   - SUPABASE_API_KEY (optional)
echo   - ALLOWED_ORIGINS (Frontend URL)
echo.

pause
