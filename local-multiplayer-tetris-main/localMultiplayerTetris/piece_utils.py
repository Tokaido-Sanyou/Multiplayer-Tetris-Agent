"""
Utility functions for Tetris pieces
"""

def valid_space(shape, grid):
    accepted_pos = [(j,i) for j in range(10) for i in range(20) if grid[i][j] == (0,0,0)]
    print("Accepted positions:", accepted_pos)  # Debug: Log accepted positions
    formatted = convert_shape_format(shape)
    print("Formatted positions:", formatted)  # Debug: Log formatted positions

    for pos in formatted:
        x, y = pos
        if x < 0 or x >= 10 or y >= 20:
            return False
        if y >= 0:  # Allow negative y for pieces spawning above the grid
            if pos not in accepted_pos:
                if y < 20 and grid[y][x] == (128, 128, 128):
                    return False
                return False
    return True

def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % (len(shape.shape))]

    for i, line in enumerate(format):
        for j, column in enumerate(line):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    # No UI offset here; raw grid positions returned
    for i,pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)


    return positions