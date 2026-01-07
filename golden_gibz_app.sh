#!/bin/bash

# Golden Gibz Trading System Launcher Script
# For Linux and macOS users

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "  ========================================"
echo "   🤖 Golden Gibz Trading System 🤖"
echo "  ========================================"
echo -e "${NC}"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Error: Python is not installed!${NC}"
        echo "Please install Python 3.7+ and try again."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo -e "${GREEN}✅ Python found: $($PYTHON_CMD --version)${NC}"

# Check if we're in the right directory
if [ ! -f "golden_gibz_app.py" ]; then
    echo -e "${RED}❌ Error: golden_gibz_app.py not found!${NC}"
    echo "Please ensure you're running this from the correct directory."
    exit 1
fi

echo -e "${GREEN}✅ Application files found!${NC}"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}🔍 Virtual environment found, activating...${NC}"
    source venv/bin/activate
fi

# Launch the application
echo -e "${BLUE}🚀 Starting Golden Gibz Trading System...${NC}"
echo ""

$PYTHON_CMD launch_golden_gibz_app.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Application closed successfully.${NC}"
else
    echo ""
    echo -e "${RED}❌ Application encountered an error.${NC}"
    echo "Please check the error messages above."
    exit 1
fi