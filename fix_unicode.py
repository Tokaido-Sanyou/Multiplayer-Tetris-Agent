#!/usr/bin/env python3
"""
Fix Unicode emoji characters in training files
"""

import re

def fix_file(filename):
    """Remove emoji characters from file"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace common emojis with text
    content = content.replace('📊', '')
    content = content.replace('🔗', '')
    content = content.replace('🎉', '')
    content = content.replace('✅', '')
    content = content.replace('🏃', '')
    content = content.replace('🤖', '')
    content = content.replace('⏹️', '')
    content = content.replace('🚀', '')
    content = content.replace('🎯', '')
    content = content.replace('⚠️', '')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed emojis in {filename}")

if __name__ == "__main__":
    files = [
        'train_dqn_hierarchical.py',
        'train_dqn_movement.py',
        'train_dream.py'
    ]
    
    for file in files:
        try:
            fix_file(file)
        except Exception as e:
            print(f"Error fixing {file}: {e}") 