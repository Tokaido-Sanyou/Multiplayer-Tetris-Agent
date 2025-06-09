@echo off
echo Starting TensorBoard for DQN Movement Agent...
echo Access at: http://localhost:6009
python -m tensorboard.main --logdir=logs/dqn_movement_standard/tensorboard --port=6009 