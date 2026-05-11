# PowerShell script to run Clinical Lab Analysis API
# Equivalent to run_api.sh for Windows

Write-Host "=================================" -ForegroundColor Green
Write-Host "Clinical Lab Analysis API" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Check if venv exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install/upgrade requirements
Write-Host "Installing/upgrading dependencies..." -ForegroundColor Yellow
pip install --upgrade -r requirements.txt

# Create logs directory
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# Check if models exist (check for any agent2 model)
$agent2Models = @("agent2.pkl", "agent2_best_model.pkl", "agent2_model.pkl")
$modelExists = $false
foreach ($model in $agent2Models) {
    $modelPath = "backend/ml/models/$model"
    if (Test-Path $modelPath) {
        $modelExists = $true
        break
    }
}

if (-not $modelExists) {
    Write-Host "ERROR: Agent 2 model not found!" -ForegroundColor Red
    Write-Host "Please run Week 2 training first: python backend/ml/train_production.py"
    Write-Host "Expected one of: $($agent2Models -join ', ')"
    exit 1
}

# Start API
if ($args[0] -eq "production") {
    Write-Host "Starting API in PRODUCTION mode with Gunicorn..." -ForegroundColor Green
    gunicorn app.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 4 --access-logfile logs/access.log --error-logfile logs/error.log
} else {
    Write-Host "Starting API in DEVELOPMENT mode with Uvicorn..." -ForegroundColor Green
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
}