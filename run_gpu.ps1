# ============================================================
# BUS-XAINet GPU Training Launcher
# Runs training with Python 3.10 (CUDA-enabled build)
# Usage: Right-click -> Run with PowerShell, OR:
#        PS E:\CONFERENCE> .\run_gpu.ps1
# ============================================================

param (
    [switch]$NoInteractive,
    [int]$NumWorkers = 0,
    [double]$PElastic = 0.15
)

$PY = "C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe"
$ROOT = $PSScriptRoot

# Verify GPU is available
Write-Host "`n=== Checking GPU... ===" -ForegroundColor Cyan
& $PY -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

$EXTRA_ARGS = @()
if ($NoInteractive) {
    $EXTRA_ARGS += "--no-interactive"
}
if ($NumWorkers -gt 0) {
    $EXTRA_ARGS += "--num-workers", $NumWorkers
}
if ($PElastic -ne 0.15) {
    $EXTRA_ARGS += "--p-elastic", $PElastic
}

Write-Host "`n=== Starting Stage 1: U-Net Segmentation Training (40 epochs) ===" -ForegroundColor Green
& $PY -u "$ROOT\training\train_unet.py" $EXTRA_ARGS

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ABORTED] U-Net training failed or user rejected masking." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Starting Stage 2: BUS-XAINet Classifier Training (40 epochs x 5 folds) ===" -ForegroundColor Green
& $PY -u "$ROOT\training\train_classifier.py" $EXTRA_ARGS

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ABORTED] Classifier training failed or user rejected performance." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Full Pipeline Completed Successfully! ===" -ForegroundColor Cyan
Write-Host "Models saved to: $ROOT\outputs\models\" -ForegroundColor Yellow
