#!/usr/bin/env python3
"""
TensorBoard Launcher Script
Simple script to launch TensorBoard for Tetris RL training logs
"""

import sys
from tensorboard import main

def launch_tensorboard(logdir='logs/unified_training', port=6006):
    """Launch TensorBoard with specified log directory"""
    print(f"🚀 Starting TensorBoard...")
    print(f"📁 Log directory: {logdir}")
    print(f"🌐 Port: {port}")
    print(f"🔗 URL: http://localhost:{port}")
    print("Press Ctrl+C to stop TensorBoard")
    
    # Launch TensorBoard
    main.run_main([
        '--logdir', logdir,
        '--port', str(port),
        '--host', '0.0.0.0'
    ])

if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) > 1:
        logdir = sys.argv[1]
    else:
        logdir = 'logs/unified_training'
    
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    else:
        port = 6006
    
    launch_tensorboard(logdir, port) 