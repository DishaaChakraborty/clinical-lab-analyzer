#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Clinical Lab Analysis API${NC}"
echo -e "${GREEN}================================${NC}"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install/upgrade requirements
echo -e "${YELLOW}Installing/upgrading dependencies...${NC}"
pip install --upgrade -r requirements.txt

# Create logs directory
mkdir -p logs

# Check if models exist
if [ ! -f "backend/ml/models/agent2_best_model.pkl" ]; then
    echo -e "${RED}ERROR: Models not found!${NC}"
    echo "Please run Week 2 training first: python backend/ml/train_production.py"
    exit 1
fi

# Start API
if [ "$1" = "production" ]; then
    echo -e "${GREEN}Starting API in PRODUCTION mode with Gunicorn...${NC}"
    gunicorn app.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 4 --access-logfile logs/access.log --error-logfile logs/error.log
else
    echo -e "${GREEN}Starting API in DEVELOPMENT mode with Uvicorn...${NC}"
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
fi