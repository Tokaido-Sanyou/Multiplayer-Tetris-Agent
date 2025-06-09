@echo off
REM TensorBoard Commands for All Trainers
REM Usage: Run each command in a separate terminal window

echo TensorBoard Commands for Tetris Agents
echo ==========================================
echo.
echo DQN Locked Agent:
echo python -m tensorboard.main --logdir=logs/dqn_locked_standard/tensorboard --port=6007
echo.
echo DQN Movement Agent:
echo python -m tensorboard.main --logdir=logs/dqn_movement_standard/tensorboard --port=6009
echo.
echo DQN Hierarchical Agent:
echo python -m tensorboard.main --logdir=logs/dqn_hierarchical_standard/tensorboard --port=6008
echo.
echo DREAM Agent:
echo python -m tensorboard.main --logdir=logs/dream_fixed_complete/tensorboard --port=6006
echo.
echo ==========================================
echo Access URLs:
echo DQN Locked:     http://localhost:6007
echo DQN Movement:   http://localhost:6009
echo DQN Hierarchical: http://localhost:6008
echo DREAM:          http://localhost:6006
echo ========================================== 