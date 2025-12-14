@echo off
echo ========================================
echo Complete Dependency Download for Offline Installation
echo ========================================
echo.

:: Create directories
if not exist "dependencies\wheels" mkdir dependencies\wheels
if not exist "dependencies\torch" mkdir dependencies\torch

echo Downloading ALL currently installed packages for offline use...
echo.

:: Core Python packages
echo [1/8] Downloading core Python packages...
pip download -d dependencies\wheels ^
    certifi ^
    charset-normalizer ^
    idna ^
    requests ^
    urllib3 ^
    setuptools ^
    pip ^
    wheel

:: Scientific computing packages
echo [2/8] Downloading scientific computing packages...
pip download -d dependencies\wheels ^
    numpy ^
    pandas ^
    scipy ^
    scikit-learn ^
    matplotlib ^
    seaborn ^
    statsmodels

:: Technical analysis packages
echo [3/8] Downloading technical analysis packages...
pip download -d dependencies\wheels ^
    pandas-ta ^
    numba ^
    llvmlite

:: Reinforcement learning packages
echo [4/8] Downloading RL packages...
pip download -d dependencies\wheels ^
    stable-baselines3 ^
    gymnasium ^
    optuna ^
    tensorboard ^
    tensorboard-data-server

:: Utility packages
echo [5/8] Downloading utility packages...
pip download -d dependencies\wheels ^
    tqdm ^
    pyyaml ^
    joblib ^
    cloudpickle ^
    psutil ^
    packaging ^
    typing-extensions

:: Visualization and plotting
echo [6/8] Downloading visualization packages...
pip download -d dependencies\wheels ^
    matplotlib ^
    seaborn ^
    pillow ^
    contourpy ^
    cycler ^
    fonttools ^
    kiwisolver ^
    pyparsing

:: Database and configuration
echo [7/8] Downloading database and config packages...
pip download -d dependencies\wheels ^
    sqlalchemy ^
    alembic ^
    greenlet ^
    mako ^
    colorlog

:: Additional packages found in current installation
echo [8/8] Downloading additional packages...
pip download -d dependencies\wheels ^
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
    absl-py ^
    mutagen ^
    pefile ^
    pycryptodomex ^
    pyinstaller ^
    pyinstaller-hooks-contrib ^
    pywin32-ctypes ^
    websockets ^
    yt-dlp

echo.
echo Downloading PyTorch (CPU and GPU versions)...
pip download -d dependencies\torch ^
    torch ^
    torchvision ^
    torchaudio ^
    --index-url https://download.pytorch.org/whl/cpu

echo.
echo Downloading PyTorch GPU version (CUDA 11.8)...
pip download -d dependencies\torch ^
    torch ^
    torchvision ^
    torchaudio ^
    --index-url https://download.pytorch.org/whl/cu118

echo.
echo Creating comprehensive requirements file...
pip freeze > dependencies\requirements_complete.txt

echo.
echo ========================================
echo COMPLETE DOWNLOAD FINISHED!
echo ========================================
echo.
echo Downloaded packages:
dir dependencies\wheels\*.whl | find /c ".whl"
echo.
echo PyTorch packages:
dir dependencies\torch\*.whl | find /c ".whl"
echo.
echo Total size:
for /f %%i in ('dir dependencies /s /-c ^| find "bytes"') do echo %%i
echo.
echo To install offline: dependencies\install_complete_offline.bat
echo.