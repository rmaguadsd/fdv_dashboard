# Start Ollama with models directory set to D:\FDV\models
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\FDV\models", "User")
$env:OLLAMA_MODELS = "D:\FDV\models"

# Kill any existing Ollama processes
Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# Start Ollama
Write-Host "Starting Ollama with OLLAMA_MODELS=D:\FDV\models" -ForegroundColor Cyan
Start-Process ollama -WindowStyle Normal -ArgumentList "serve"
Start-Sleep -Seconds 5
Write-Host "Ollama started. Models directory: D:\FDV\models" -ForegroundColor Green
