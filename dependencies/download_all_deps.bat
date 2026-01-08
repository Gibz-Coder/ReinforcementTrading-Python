@echo off
echo ========================================
echo Golden-Gibz Dependency Downloader
echo ========================================
echo.
echo This will download all required packages for offline installation.
echo Make sure you have a good internet connection.
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

echo ✅ Python found
echo.

:: Check pip
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip not found
    echo Please ensure pip is installed
    pause
    exit /b 1
)

echo ✅ pip found
echo.

:: Upgrade pip first
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

:: Run the downloader
echo.
echo 🚀 Starting download...
python dependencies\download_all_deps.py

if errorlevel 1 (
    echo.
    echo ❌ Download failed
    echo Check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ✅ Download completed successfully!
echo.
echo You can now use offline installation:
echo   dependencies\smart_install.bat
echo.
pause