@echo off
echo Starting TensorBoard for DQN Hierarchical Agent...
echo Access at: http://localhost:6008
python -m tensorboard.main --logdir=logs/dqn_hierarchical_standard/tensorboard --port=6008 