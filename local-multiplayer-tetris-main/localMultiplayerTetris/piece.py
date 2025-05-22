from constants import I, O, JLSTZ_WALL_KICKS, I_WALL_KICKS, shapes, shape_colors
from utils import valid_space

class Piece:
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.rotation = 0
        self.color = shape_colors[shapes.index(shape)]
    
    def rotate(self, direction, grid):
        """Rotate the piece with SRS wall kicks.
        direction: 1 for clockwise, -1 for counter-clockwise
        Returns True if rotation was successful, False otherwise"""
        old_rotation = self.rotation
        new_rotation = (self.rotation + direction) % len(self.shape)
        
        # Get the appropriate wall kick data
        if self.shape == I:
            wall_kicks = I_WALL_KICKS
        elif self.shape == O:
            # O piece doesn't need wall kicks
            self.rotation = new_rotation
            return True
        else:
            wall_kicks = JLSTZ_WALL_KICKS
        
        # Try each wall kick offset
        for x_offset, y_offset in wall_kicks[old_rotation][new_rotation]:
            self.x += x_offset
            self.y += y_offset
            self.rotation = new_rotation
            
            if valid_space(self, grid):
                return True
            
            # If the position is invalid, revert the changes
            self.x -= x_offset
            self.y -= y_offset
            self.rotation = old_rotation
        
        return False 