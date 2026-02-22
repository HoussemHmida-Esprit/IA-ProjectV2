# Test Deployment Script for PowerShell
# Tests the deployed backend on Render

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Testing Backend Deployment" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

$baseUrl = "https://ia-projectv2.onrender.com"

# Test 1: Health Check
Write-Host "Test 1: Health Check" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/api/health" -Method Get
    Write-Host "✅ Backend is online" -ForegroundColor Green
    Write-Host "Models loaded: $($health.models_loaded)" -ForegroundColor White
    Write-Host "XGBoost available: $($health.models.xgboost)" -ForegroundColor White
    Write-Host ""
} catch {
    Write-Host "❌ Health check failed: $_" -ForegroundColor Red
    Write-Host ""
    exit 1
}

# Test 2: Prediction
Write-Host "Test 2: Prediction Request" -ForegroundColor Yellow
$predictionBody = @{
    lighting = 1
    location = 2
    intersection = 1
    day_of_week = 3
    hour = 14
    num_users = 2
    model = "xgboost"
} | ConvertTo-Json

try {
    $prediction = Invoke-RestMethod -Uri "$baseUrl/api/predict" -Method Post -ContentType "application/json" -Body $predictionBody
    Write-Host "✅ Prediction successful" -ForegroundColor Green
    Write-Host "Collision: $($prediction.final_prediction.collision.class_name)" -ForegroundColor White
    Write-Host "Severity: $($prediction.final_prediction.severity.class_name)" -ForegroundColor White
    Write-Host "Confidence: $([math]::Round($prediction.final_prediction.collision.confidence * 100, 2))%" -ForegroundColor White
    Write-Host ""
} catch {
    Write-Host "❌ Prediction failed: $_" -ForegroundColor Red
    Write-Host ""
}

# Test 3: Available Models
Write-Host "Test 3: Available Models" -ForegroundColor Yellow
try {
    $models = Invoke-RestMethod -Uri "$baseUrl/api/models" -Method Get
    Write-Host "✅ Models endpoint working" -ForegroundColor Green
    Write-Host "Available models: $($models.models.Count)" -ForegroundColor White
    foreach ($model in $models.models) {
        Write-Host "  - $($model.name)" -ForegroundColor White
    }
    Write-Host ""
} catch {
    Write-Host "❌ Models endpoint failed: $_" -ForegroundColor Red
    Write-Host ""
}

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Testing Complete" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
