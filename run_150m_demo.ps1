<#
Run the recovered 150M Anarkali model locally.

From the repository root:
  powershell -ExecutionPolicy Bypass -File .\run_150m_demo.ps1
#>
param(
    [switch] $Cpu
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$checkpoint = Join-Path $root 'anarkali-150m-best.pt'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python was not found at $python"
}
if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw "Recovered model was not found at $checkpoint"
}

& $python -c 'import torch, transformers' 2>$null
$ready = $LASTEXITCODE -eq 0
if (-not $ready) {
    Write-Host 'Installing the model runtime (first run only)...'
    & $python -m pip install --upgrade 'torch>=2.5,<3' 'transformers>=4.48,<6' 'safetensors>=0.4'
    if ($LASTEXITCODE -ne 0) {
        throw 'Model runtime installation failed.'
    }
}

if ($Cpu) {
    $device = 'cpu'
} else {
    & $python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'
    $device = if ($LASTEXITCODE -eq 0) { 'cuda' } else { 'cpu' }
}

Write-Host "Running recovered Ettin 150M on $device..."
& $python (Join-Path $root 'scripts\demo_recovered_checkpoint.py') --checkpoint $checkpoint --device $device
