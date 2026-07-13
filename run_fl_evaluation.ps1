# Run Federated Learning Evaluation
# This script evaluates federated learning performance

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Federated Learning Evaluation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run FL on CIFAR10 with IID partitioning
Write-Host "Running FL on CIFAR10 (IID, 10 clients)..." -ForegroundColor Yellow
python purifier.py --dataset cifar10 --federated --num-clients 10 --local-epochs 5 --global-rounds 10

Write-Host ""
Write-Host "Running FL on CIFAR10 (Non-IID, alpha=0.5, 10 clients)..." -ForegroundColor Yellow
python purifier.py --dataset cifar10 --federated --num-clients 10 --non-iid --alpha 0.5 --local-epochs 5 --global-rounds 10

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "FL Evaluation Complete!" -ForegroundColor Green
Write-Host "Results saved to e_purifier_fl_results_*.json" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
