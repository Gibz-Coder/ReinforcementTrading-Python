@echo off
title Golden Gibz Trading System Launcher
color 0A

echo.
echo  ========================================
echo   🤖 Golden Gibz Trading System 🤖
echo  ========================================
echo.
echo  Starting application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH!
    echo Please install Python 3.7+ and try again.
    echo.
    pause
    exit /b 1
)

REM Check if the main application file exists
if not exist "golden_gibz_app.py" (
    echo ❌ Error: golden_gibz_app.py not found!
    echo Please ensure you're running this from the correct directory.
    echo.
    pause
    exit /b 1
)

REM Launch the application
echo ✅ Python found! Launching Golden Gibz...
echo.
python launch_golden_gibz_app.py

REM Check if the application ran successfully
if errorlevel 1 (
    echo.
    echo ❌ Application encountered an error.
    echo Please check the error messages above.
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Application closed successfully.
echo.
pause