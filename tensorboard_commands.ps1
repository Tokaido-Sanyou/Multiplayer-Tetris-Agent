# TensorBoard Commands for All Trainers - PowerShell Version
# Usage: Run each command in a separate PowerShell terminal

Write-Host "TensorBoard Commands for Tetris Agents" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

Write-Host "DQN Locked Agent:" -ForegroundColor Yellow
Write-Host "python -m tensorboard.main --logdir=logs/dqn_locked_standard/tensorboard --port=6007" -ForegroundColor Cyan
Write-Host ""

Write-Host "DQN Movement Agent:" -ForegroundColor Yellow  
Write-Host "python -m tensorboard.main --logdir=logs/dqn_movement_standard/tensorboard --port=6009" -ForegroundColor Cyan
Write-Host ""

Write-Host "DQN Hierarchical Agent:" -ForegroundColor Yellow
Write-Host "python -m tensorboard.main --logdir=logs/dqn_hierarchical_standard/tensorboard --port=6008" -ForegroundColor Cyan
Write-Host ""

Write-Host "DREAM Agent:" -ForegroundColor Yellow
Write-Host "python -m tensorboard.main --logdir=logs/dream_fixed_complete/tensorboard --port=6006" -ForegroundColor Cyan
Write-Host ""

Write-Host "==========================================" -ForegroundColor Green
Write-Host "Access URLs:" -ForegroundColor Green
Write-Host "DQN Locked:      http://localhost:6007" -ForegroundColor White
Write-Host "DQN Movement:    http://localhost:6009" -ForegroundColor White  
Write-Host "DQN Hierarchical: http://localhost:6008" -ForegroundColor White
Write-Host "DREAM:           http://localhost:6006" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Green 