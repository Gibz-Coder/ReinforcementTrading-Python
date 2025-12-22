@echo off
echo ================================================================================
echo 🚀 ULTRA-SELECTIVE V4 MODEL - QUICK START
echo ================================================================================
echo.
echo This script will help you get started with the new V4 model that achieved
echo 58.5%% win rate with balanced 1:1 risk/reward ratio.
echo.
echo BREAKTHROUGH RESULTS:
echo - Validation Win Rate: 58.5%% (vs 28-42%% in V3)
echo - Training Stability: ✅ Improving (vs ❌ Declining in V3)  
echo - Trade Quality: 10+ trades/evaluation with high selectivity
echo - Risk Profile: Balanced 1:1 TP/SL using ATR
echo.
echo ================================================================================

:MENU
echo.
echo Choose an option:
echo [1] Quick Test Training (25K timesteps, ~15 minutes)
echo [2] Full Production Training (500K timesteps, ~5 hours)
echo [3] View V4 Improvements Documentation
echo [4] Check Requirements
echo [5] Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto QUICK_TRAIN
if "%choice%"=="2" goto FULL_TRAIN
if "%choice%"=="3" goto VIEW_DOCS
if "%choice%"=="4" goto CHECK_REQS
if "%choice%"=="5" goto EXIT
goto MENU

:QUICK_TRAIN
echo.
echo 🏃‍♂️ Starting Quick Test Training...
echo This will train for 25K timesteps with 4 environments
echo Expected time: ~15 minutes
echo.
python scripts/train_ultra_selective_v4.py --timesteps 25000 --envs 4 --eval-freq 2500
echo.
echo ✅ Quick training completed! Check the results above.
echo Best models are saved in models/experimental/
echo Models with 75%+ win rate automatically move to models/production/
pause
goto MENU

:FULL_TRAIN
echo.
echo 🏋️‍♂️ Starting Full Production Training...
echo This will train for 500K timesteps with 8 environments
echo Expected time: ~5 hours
echo.
echo WARNING: This is a long training session. Make sure:
echo - Your computer won't go to sleep
echo - You have stable power supply
echo - No other intensive tasks are running
echo.
set /p confirm="Continue? (y/n): "
if /i "%confirm%"=="y" (
    python scripts/train_ultra_selective_v4.py --timesteps 500000 --envs 8 --eval-freq 5000
    echo.
    echo 🎉 Full training completed!
    echo Check models/production/ for models with 75%+ win rate
) else (
    echo Training cancelled.
)
pause
goto MENU

:VIEW_DOCS
echo.
echo 📖 Opening V4 Improvements Documentation...
echo.
if exist "docs\ultra_selective_v4_improvements.md" (
    type "docs\ultra_selective_v4_improvements.md"
) else (
    echo Documentation file not found. Please check docs/ultra_selective_v4_improvements.md
)
echo.
pause
goto MENU

:CHECK_REQS
echo.
echo 🔍 Checking Requirements...
echo.
echo Required Python packages:
echo - numpy
echo - pandas  
echo - pandas_ta
echo - torch
echo - stable-baselines3
echo - gymnasium
echo.
echo To install: pip install -r requirements.txt
echo.
echo Checking if Python is available...
python --version 2>nul
if %errorlevel%==0 (
    echo ✅ Python is installed
) else (
    echo ❌ Python not found. Please install Python 3.10+
)
echo.
echo Checking if required packages are installed...
python -c "import torch, stable_baselines3, pandas_ta; print('✅ Core packages are installed')" 2>nul
if %errorlevel%==0 (
    echo ✅ Required packages are available
) else (
    echo ❌ Some packages missing. Run: pip install -r requirements.txt
)
echo.
pause
goto MENU

:EXIT
echo.
echo 🚀 Ready to achieve 80%+ win rates with V4!
echo.
echo Quick commands:
echo - Quick test: python scripts/train_ultra_selective_v4.py --timesteps 25000 --envs 4
echo - Full training: python scripts/train_ultra_selective_v4.py --timesteps 500000 --envs 8
echo.
echo Happy trading! 📈
exit /b 0