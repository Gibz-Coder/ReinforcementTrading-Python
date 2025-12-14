@echo off
echo ========================================
echo Complete Offline Installation System
echo ========================================
echo.

:: Check if we're in a virtual environment
if "%VIRTUAL_ENV%"=="" (
    echo ERROR: Please activate virtual environment first
    echo Run: forex_env\Scripts\activate.bat
    pause
    exit /b 1
)

echo Installing ALL packages from local wheels...
echo This may take several minutes...
echo.

:: Upgrade pip first
echo [1/6] Upgrading pip...
python -m pip install --upgrade pip

:: Install core packages first (order matters for dependencies)
echo [2/6] Installing core Python packages...
pip install --no-index --find-links wheels\ --force-reinstall ^
    setuptools ^
    wheel ^
    certifi ^
    charset-normalizer ^
    idna ^
    urllib3 ^
    requests

if errorlevel 1 (
    echo ERROR: Failed to install core packages
    pause
    exit /b 1
)

:: Install scientific computing packages
echo [3/6] Installing scientific computing packages...
pip install --no-index --find-links wheels\ --force-reinstall ^
    numpy ^
    scipy ^
    pandas ^
    scikit-learn ^
    matplotlib ^
    seaborn

if errorlevel 1 (
    echo ERROR: Failed to install scientific packages
    pause
    exit /b 1
)

:: Install PyTorch (try CPU version first, then GPU if available)
echo [4/6] Installing PyTorch...
if exist "torch\torch-*cpu*.whl" (
    echo Installing PyTorch CPU version...
    pip install --no-index --find-links torch\ --force-reinstall torch torchvision torchaudio
) else (
    echo PyTorch wheels not found, installing from PyPI...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

:: Install RL and trading packages
echo [5/6] Installing RL and trading packages...
pip install --no-index --find-links wheels\ --force-reinstall ^
    stable-baselines3 ^
    gymnasium ^
    pandas-ta ^
    numba ^
    llvmlite ^
    optuna ^
    tensorboard ^
    tensorboard-data-server

if errorlevel 1 (
    echo ERROR: Failed to install RL packages
    pause
    exit /b 1
)

:: Install all remaining packages
echo [6/6] Installing remaining packages...
pip install --no-index --find-links wheels\ --force-reinstall ^
    tqdm ^
    pyyaml ^
    joblib ^
    cloudpickle ^
    psutil ^
    packaging ^
    typing-extensions ^
    pillow ^
    contourpy ^
    cycler ^
    fonttools ^
    kiwisolver ^
    pyparsing ^
    sqlalchemy ^
    alembic ^
    greenlet ^
    mako ^
    colorlog ^
    brotli ^
    altgraph ^
    farama-notifications ^
    filelock ^
    fsspec ^
    grpcio ^
    jinja2 ^
    markdown ^
    markupsafe ^
    mpmath ^
    networkx ^
    protobuf ^
    sympy ^
    werkzeug ^
    absl-py

:: Install optional packages (don't fail if these don't work)
echo Installing optional packages...
pip install --no-index --find-links wheels\ --no-deps --ignore-installed ^
    mutagen ^
    pefile ^
    pycryptodomex ^
    pyinstaller ^
    pyinstaller-hooks-contrib ^
    pywin32-ctypes ^
    websockets ^
    yt-dlp 2>nul

echo.
echo ========================================
echo COMPLETE OFFLINE INSTALLATION FINISHED!
echo ========================================
echo.
echo Verifying installation...
python -c "import torch, stable_baselines3, pandas, numpy, sklearn, matplotlib; print('✓ Core packages working')"
python -c "import pandas_ta, optuna, tensorboard; print('✓ Trading packages working')"
python -c "import gymnasium, numba; print('✓ RL packages working')"

echo.
echo Installed packages summary:
pip list | find /c " "
echo.
echo Installation complete! You can now run:
echo   python scripts\train_ultra_aggressive.py
echo   python scripts\verify_installation.py
echo.