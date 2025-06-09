@echo off
echo Starting TensorBoard for DREAM Agent...
echo Access at: http://localhost:6006
python -m tensorboard.main --logdir=logs/dream_fixed_complete/tensorboard --port=6006 