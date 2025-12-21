@echo off
echo ============================================
echo   Deploy AI Trading EA to MT5
echo ============================================
echo.

REM === CONFIGURE THESE PATHS FOR YOUR SYSTEM ===
set MT5_DATA=C:\Users\Visual Coder\AppData\Roaming\MetaQuotes\Terminal\29E91DA909EB4475AB204481D1C2CE7D
REM =============================================

set SOURCE_EA=mt5_ea\HighWinRateEA_v2.mq5
set SOURCE_ONNX=mt5_export\trading_model.onnx

set DEST_EA=%MT5_DATA%\MQL5\Experts\RF Trading
set DEST_FILES=%MT5_DATA%\MQL5\Files
set DEST_TESTER=C:\Users\Visual Coder\AppData\Roaming\MetaQuotes\Tester\29E91DA909EB4475AB204481D1C2CE7D\Agent-127.0.0.1-3000\MQL5\Files

echo Checking source files...
if not exist "%SOURCE_EA%" (
    echo ERROR: EA not found: %SOURCE_EA%
    pause
    exit /b 1
)
if not exist "%SOURCE_ONNX%" (
    echo ERROR: ONNX not found: %SOURCE_ONNX%
    pause
    exit /b 1
)
echo OK
echo.

echo Creating directories...
if not exist "%DEST_EA%" mkdir "%DEST_EA%"
if not exist "%DEST_FILES%" mkdir "%DEST_FILES%"
if not exist "%DEST_TESTER%" mkdir "%DEST_TESTER%"

echo.
echo Copying files...
echo [1/3] EA to Experts folder...
copy /Y "%SOURCE_EA%" "%DEST_EA%\"

echo [2/3] ONNX to MQL5\Files (for resource embedding)...
copy /Y "%SOURCE_ONNX%" "%DEST_FILES%\"

echo [3/3] ONNX to Tester folder...
copy /Y "%SOURCE_ONNX%" "%DEST_TESTER%\"

echo.
echo ============================================
echo   DEPLOYMENT COMPLETE
echo ============================================
echo.
echo Next steps:
echo   1. Open MT5 Terminal
echo   2. Press F4 to open MetaEditor
echo   3. Navigate to Experts\RF Trading\HighWinRateEA_v2.mq5
echo   4. Press F7 to compile
echo   5. Attach EA to XAUUSD M15 chart
echo.
echo Settings:
echo   - TP: 0.5 x ATR (small target for high win rate)
echo   - SL: 2.0 x ATR (larger stop loss)
echo   - Lot Size: 0.01 (adjust based on account)
echo   - Max Daily Trades: 20
echo.
pause
