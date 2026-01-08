@echo off
echo ========================================
echo Golden-Gibz Smart Dependency Installer
echo ========================================
echo.

:: Check if we're in a virtual environment
if "%VIRTUAL_ENV%"=="" (
    echo ERROR: Please activate virtual environment first
    echo.
    echo To create and activate virtual environment:
    echo   python -m venv forex_env
    echo   forex_env\Scripts\activate.bat
    echo.
    echo Then run this installer again.
    pause
    exit /b 1
)

echo ✅ Virtual environment detected: %VIRTUAL_ENV%
echo.

:: Check Python version
python --version
if errorlevel 1 (
    echo ERROR: Python not found or not working
    pause
    exit /b 1
)

echo.
echo 🚀 Starting smart dependency installation...
echo This will:
echo   1. Check local dependencies folder
echo   2. Download missing packages if needed
echo   3. Install everything offline when possible
echo   4. Verify installation
echo.

:: Run the smart installer
python dependencies\smart_install.py

if errorlevel 1 (
    echo.
    echo ❌ Smart installation failed
    echo Trying fallback installation...
    echo.
    
    :: Fallback to basic pip install
    echo Installing core packages from PyPI...
    pip install --upgrade pip setuptools wheel
    pip install numpy pandas matplotlib scikit-learn
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install stable-baselines3 gymnasium
    pip install ta pandas-ta PyYAML tqdm colorama psutil
    pip install MetaTrader5
    
    if errorlevel 1 (
        echo ❌ Fallback installation also failed
        echo.
        echo Please check your internet connection and try again.
        echo Or install packages manually:
        echo   pip install -r requirements.txt
        pause
        exit /b 1
    )
    
    echo ✅ Fallback installation completed
)

echo.
echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.

:: Final verification
echo 🔍 Final verification...
python -c "import numpy, pandas, torch, stable_baselines3, gymnasium; print('✅ All core packages working!')" 2>nul
if errorlevel 1 (
    echo ⚠️ Some packages may not be working correctly
    echo Try running: python test_training_setup.py
) else (
    echo ✅ All packages verified successfully!
)

echo.
echo You can now run:
echo   python golden_gibz_native_app.py
echo   python train_golden_gibz_model.py
echo.
pause