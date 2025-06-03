#!/usr/bin/env python3
"""
Script to fix relative imports in all Python files
"""
import os
import re

def fix_imports_in_file(filepath):
    """Fix relative imports in a single file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip if already has try/except block
    if 'Handle both direct execution and module import' in content:
        return False
    
    # Find all relative imports
    relative_imports = re.findall(r'from\s+\.[^\s]+\s+import[^\n]+', content)
    
    if not relative_imports:
        return False
    
    # Create the try/except block
    try_imports = []
    except_imports = []
    
    for imp in relative_imports:
        try_imports.append(imp)
        # Convert relative to absolute
        absolute_imp = imp.replace('from .', 'from ')
        except_imports.append(absolute_imp)
    
    # Remove existing relative imports
    for imp in relative_imports:
        content = content.replace(imp + '\n', '')
    
    # Find insertion point (after other imports)
    lines = content.split('\n')
    insert_idx = 0
    
    # Find last import line
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insert_idx = i + 1
    
    # Insert try/except block
    new_imports = [
        '',
        '# Handle both direct execution and module import',
        'try:'
    ] + ['    ' + imp for imp in try_imports] + [
        'except ImportError:',
        '    # Direct execution - imports without relative paths'
    ] + ['    ' + imp for imp in except_imports]
    
    lines[insert_idx:insert_idx] = new_imports
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return True

def main():
    """Fix imports in all Python files"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fixed_count = 0
    
    for root, dirs, files in os.walk(base_dir):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py') and file != 'fix_imports.py':
                filepath = os.path.join(root, file)
                if fix_imports_in_file(filepath):
                    print(f"Fixed imports in: {filepath}")
                    fixed_count += 1
    
    print(f"\nFixed imports in {fixed_count} files")

if __name__ == '__main__':
    main() 