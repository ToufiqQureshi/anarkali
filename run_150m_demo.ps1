<#
Run the released 150M Anarkali ONNX model locally.

From the repository root:
  powershell -ExecutionPolicy Bypass -File .\run_150m_demo.ps1
#>
param(
    [string] $Request = "examples\requests\support_routing.json"
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
$model = Join-Path $root 'release-150m'
$requestPath = Join-Path $root $Request

if (-not (Test-Path -LiteralPath $model)) {
    throw "Released model directory was not found at $model"
}
if (-not (Test-Path -LiteralPath (Join-Path $model 'model.onnx'))) {
    throw "ONNX graph was not found at $model"
}
if (-not (Test-Path -LiteralPath $requestPath)) {
    throw "Request fixture was not found at $requestPath"
}

& $python -c 'import onnxruntime, tokenizers, numpy' 2>$null
$ready = $LASTEXITCODE -eq 0
if (-not $ready) {
    Write-Host 'Installing the ONNX runtime (first run only)...'
    & $python -m pip install --upgrade 'onnxruntime>=1.18' 'tokenizers>=0.19' 'numpy'
    if ($LASTEXITCODE -ne 0) {
        throw 'ONNX runtime installation failed.'
    }
}

Write-Host "Running released Anarkali 150M from $model..."
& $python -m anarkali decide --model $model --request $requestPath
