@echo off
echo Starting All TensorBoard Instances...
echo This will open 4 separate terminal windows

start "DQN Locked TensorBoard" cmd /k "python -m tensorboard.main --logdir=logs/dqn_locked_standard/tensorboard --port=6007"

start "DQN Movement TensorBoard" cmd /k "python -m tensorboard.main --logdir=logs/dqn_movement_standard/tensorboard --port=6009"

start "DQN Hierarchical TensorBoard" cmd /k "python -m tensorboard.main --logdir=logs/dqn_hierarchical_standard/tensorboard --port=6008"

start "DREAM TensorBoard" cmd /k "python -m tensorboard.main --logdir=logs/dream_fixed_complete/tensorboard --port=6006"

echo.
echo All TensorBoard instances started!
echo Access URLs:
echo DQN Locked:     http://localhost:6007
echo DQN Movement:   http://localhost:6009
echo DQN Hierarchical: http://localhost:6008
echo DREAM:          http://localhost:6006
echo.
echo Press any key to exit...
pause 