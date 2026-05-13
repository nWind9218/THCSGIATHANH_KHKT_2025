@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set PYTHONPATH=%cd%
"%~dp0.venv\Scripts\python.exe" api\main.py
pause
