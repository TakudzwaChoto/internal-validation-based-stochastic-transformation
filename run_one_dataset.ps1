# Run ONE dataset:  .\run_one_dataset.ps1 facescrub530
# Order: facescrub530 -> utkface -> texas100 -> purchase100 -> location -> cifar100 -> cifar10

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('facescrub530', 'utkface', 'texas100', 'purchase100', 'location', 'cifar100', 'cifar10')]
    [string]$Dataset
)

Set-Location $PSScriptRoot
$logDir = Join-Path $PSScriptRoot "eval_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$logFile = Join-Path $logDir "$Dataset.log"
Write-Host "Starting $Dataset (log: $logFile)"

$env:PYTHONIOENCODING = "utf-8"
python run_comprehensive_evaluation.py --dataset $Dataset --epochs 40 --comprehensive 2>&1 |
    Tee-Object -FilePath $logFile

exit $LASTEXITCODE
