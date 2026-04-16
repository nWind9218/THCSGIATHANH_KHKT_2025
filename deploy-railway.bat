@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================
echo    THCS GIA THANH - Railway Deployment
echo ========================================
echo.

REM Check git
where git >nul 2>&1
if %errorlevel% neq 0 (
  echo [ERROR] Git is not installed or not in PATH
  pause
  exit /b 1
)

REM Validate required files
if not exist Dockerfile (
  echo [ERROR] Dockerfile not found
  pause
  exit /b 1
)

if not exist requirements.txt (
  echo [ERROR] requirements.txt not found
  pause
  exit /b 1
)

if not exist .env.production.example (
  echo [WARNING] .env.production.example not found
)

echo [OK] Required files look good

echo.
echo [INFO] Git status:
git status --short
echo.

git add -A
git diff-index --quiet --cached HEAD
if %errorlevel% neq 0 (
  set /p COMMIT_MSG="Enter commit message (default: 'Deploy to Railway'): "
  if "!COMMIT_MSG!"=="" set COMMIT_MSG=Deploy to Railway
  git commit -m "!COMMIT_MSG!"
) else (
  echo [INFO] No changes to commit
)

echo [INFO] Pushing to GitHub...
git push -u origin main
if %errorlevel% neq 0 (
  echo [ERROR] Git push failed
  pause
  exit /b 1
)

echo.
echo [OK] Push successful
echo.
echo Next steps on Railway:
echo 1. Open https://railway.com
echo 2. Create New Project ^> Deploy from GitHub repo
echo 3. Ensure deployment uses Dockerfile
echo 4. Set environment variables:
echo    - DATABASE_URL (Supabase Postgres)
echo    - REDIS_URL (Upstash Redis, rediss://)
echo    - OPENAI_API_KEY
echo    - ALLOWED_ORIGINS
echo    - ALLOWED_METHODS
echo    - ALLOWED_HEADERS
echo    - ALLOWED_CREDENTIALS
echo    - CORS_MAX_AGE
echo    - WS_REQUIRE_ORIGIN
echo 5. Wait for service to become healthy
echo.

pause
