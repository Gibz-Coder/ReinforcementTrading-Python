@echo off
echo Installing Golden Gibz Model Training Dependencies...
echo =====================================================

echo.
echo Installing core ML/RL libraries with extra packages...
pip install "stable-baselines3[extra]>=2.0.0"
pip install gymnasium>=0.28.0
pip install torch>=1.13.0

echo.
echo Installing data processing libraries...
pip install pandas>=1.5.0
pip install numpy>=1.21.0
pip install ta>=0.10.0

echo.
echo Installing configuration and utilities...
pip install PyYAML>=6.0
pip install matplotlib>=3.5.0
pip install tensorboard>=2.10.0

echo.
echo Installing progress bar and rich formatting...
pip install tqdm>=4.64.0
pip install rich>=12.0.0

echo.
echo Installing development tools...
pip install jupyter>=1.0.0
pip install notebook>=6.4.0

echo.
echo =====================================================
echo Installation completed!
echo.
echo You can now train Golden Gibz models using:
echo   python train_golden_gibz_model.py --symbol XAUUSD --timesteps 1000000
echo.
echo Or use the Model tab in the Golden Gibz application.
echo =====================================================
pause