"""
Action handler for player controls in Tetris
"""

# Handle both direct execution and module import
try:
    from .piece_utils import valid_space, convert_shape_format
    from .utils import hard_drop, get_shape_from_index
    from .piece import Piece
except ImportError:
    # Direct execution - imports without relative paths
    from piece_utils import valid_space, convert_shape_format
    from utils import hard_drop, get_shape_from_index
    from piece import Piece

class ActionHandler:
    def __init__(self, player):
        self.player = player
        
    def move_left(self):
        """Move the current piece left"""
        if self.player.current_piece:
            self.player.current_piece.x -= 1
            from .utils import create_grid
            grid = create_grid(self.player.locked_positions)
            if not valid_space(self.player.current_piece, grid):
                self.player.current_piece.x += 1
                return False
            return True
        return False
    
    def move_right(self):
        """Move the current piece right"""
        if self.player.current_piece:
            self.player.current_piece.x += 1
            from .utils import create_grid
            grid = create_grid(self.player.locked_positions)
            if not valid_space(self.player.current_piece, grid):
                self.player.current_piece.x -= 1
                return False
            return True
        return False
    
    def move_down(self):
        """Move the current piece down"""
        if self.player.current_piece:
            self.player.current_piece.y += 1
            from .utils import create_grid
            grid = create_grid(self.player.locked_positions)
            if not valid_space(self.player.current_piece, grid):
                self.player.current_piece.y -= 1
                self.player.change_piece = True
                return False
            return True
        return False
    
    def rotate_cw(self):
        """Rotate the current piece clockwise"""
        if self.player.current_piece:
            from .utils import create_grid
            grid = create_grid(self.player.locked_positions)
            return self.player.current_piece.rotate(1, grid)
        return False
    
    def rotate_ccw(self):
        """Rotate the current piece counter-clockwise"""
        if self.player.current_piece:
            from .utils import create_grid
            grid = create_grid(self.player.locked_positions)
            return self.player.current_piece.rotate(-1, grid)
        return False
    
    def hard_drop(self):
        """Drop the current piece to the bottom"""
        if self.player.current_piece:
            from .utils import create_grid, hard_drop
            grid = create_grid(self.player.locked_positions)
            hard_drop(self.player.current_piece, grid)
            self.player.change_piece = True
            return True
        return False
    
    def hold_piece(self):
        """Hold the current piece"""
        if self.player.current_piece and self.player.can_hold:
            if self.player.hold_piece is None:
                # First hold
                self.player.hold_piece = self.player.current_piece
                self.player.current_piece = self.player.next_pieces[0]
                # Update next pieces
                self.player.current_block_index += 1
                self.player.block_pool.ensure_blocks_ahead(self.player.current_block_index)
                self.player.next_pieces = [get_shape_from_index(idx) for idx in 
                                         self.player.block_pool.get_next_blocks(self.player.current_block_index)]
            else:
                # Swap with hold piece
                temp = self.player.current_piece
                self.player.current_piece = self.player.hold_piece
                self.player.hold_piece = temp
                
                # Reset position and rotation
                self.player.current_piece.x = 5
                self.player.current_piece.y = -1
                self.player.current_piece.rotation = 0
            
            self.player.can_hold = False
            return True
        return False 