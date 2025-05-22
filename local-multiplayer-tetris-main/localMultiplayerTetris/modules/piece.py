from .constants import shapes, shape_colors, JLSTZ_WALL_KICKS, I_WALL_KICKS

class Piece:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0

    def rotate(self, grid, clockwise=True):
        # Get the current rotation state
        current_rotation = self.rotation
        
        # Calculate new rotation state
        if clockwise:
            new_rotation = (current_rotation + 1) % 4
        else:
            new_rotation = (current_rotation - 1) % 4
        
        # Get wall kick data based on piece type
        if self.shape == shapes[2]:  # I piece
            wall_kicks = I_WALL_KICKS
        else:
            wall_kicks = JLSTZ_WALL_KICKS
        
        # Get the appropriate wall kick data
        if clockwise:
            kick_data = wall_kicks[current_rotation][new_rotation]
        else:
            # For counter-clockwise, we need to use the opposite rotation
            # For example, rotating CCW from 0 to 3 is equivalent to rotating CW from 3 to 0
            kick_data = wall_kicks[new_rotation][current_rotation]
            # Reverse the x-offset for counter-clockwise rotation
            kick_data = [(-x, y) for x, y in kick_data]
        
        # Try each wall kick offset
        for offset_x, offset_y in kick_data:
            # Try the rotation with the current offset
            if self._is_valid_rotation(grid, new_rotation, offset_x, offset_y):
                self.rotation = new_rotation
                self.x += offset_x
                self.y += offset_y
                return True
        
        return False

    def _is_valid_rotation(self, grid, new_rotation, offset_x, offset_y):
        # Get the new shape after rotation
        new_shape = self.shape[new_rotation]
        
        # Check each position in the new shape
        for i, row in enumerate(new_shape):
            for j, cell in enumerate(row):
                if cell == '0':
                    # Calculate new position with offset
                    new_x = self.x + j + offset_x
                    new_y = self.y + i + offset_y
                    
                    # Check if position is valid
                    if not (0 <= new_x < len(grid[0]) and 
                           0 <= new_y < len(grid) and 
                           grid[new_y][new_x] == (0, 0, 0)):
                        return False
        return True

    def get_positions(self):
        positions = []
        for i, row in enumerate(self.shape[self.rotation]):
            for j, cell in enumerate(row):
                if cell == '0':
                    positions.append((self.x + j, self.y + i))
        return positions 