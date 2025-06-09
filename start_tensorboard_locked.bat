@echo off
echo Starting TensorBoard for DQN Locked Agent...
echo Access at: http://localhost:6007
python -m tensorboard.main --logdir=logs/dqn_locked_standard/tensorboard --port=6007 