from .piece import Piece
from .block_pool import BlockPool
from .action_handler import ActionHandler
from .key_handler import KeyHandler
from .constants import shapes
import pygame

class Player:
    def __init__(self, is_player_one=True):
        self.locked_positions = {}
        self.current_piece = None
        self.next_pieces = []
        self.hold_piece = None
        self.can_hold = True
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.is_player_one = is_player_one
        
        # Initialize block pool and get first piece
        self.block_pool = BlockPool()
        self.current_piece = self.get_new_piece()
        self.next_pieces = [self.get_new_piece() for _ in range(3)]
        
        # Initialize action and key handlers
        self.action_handler = ActionHandler(self)
        self.key_handler = KeyHandler(self.action_handler, is_player_one)

    def handle_input(self, event):
        """Handle input by passing the key to the key handler"""
        if event.type == pygame.KEYDOWN:
            self.key_handler.handle_key(event.key)

    def get_new_piece(self):
        """Get a new piece from the block pool"""
        shape_idx = self.block_pool.get_next_piece()
        return Piece(3, 0, shapes[shape_idx])

    def update(self, fall_speed):
        """Update the player's state"""
        if self.current_piece:
            if not self._is_valid_move(self.current_piece.x, self.current_piece.y + 1):
                # Lock the piece in place
                positions = self.current_piece.get_positions()
                for pos in positions:
                    self.locked_positions[pos] = self.current_piece.color
                
                # Get a new piece
                self.current_piece = self.next_pieces[0]
                self.next_pieces = self.next_pieces[1:] + [self.get_new_piece()]
                self.can_hold = True
                
                # Check for completed lines
                self._clear_lines()
                return True
            else:
                self.current_piece.y += 1
        return False

    def _is_valid_move(self, x, y):
        """Check if a move is valid"""
        if not self.current_piece:
            return False
            
        # Get the current positions of the piece
        positions = self.current_piece.get_positions()
        
        # Check each position
        for pos in positions:
            # Check if position is within bounds
            if not (0 <= pos[0] < 10 and 0 <= pos[1] < 20):
                return False
            # Check if position is already occupied
            if pos in self.locked_positions:
                return False
        return True

    def _clear_lines(self):
        """Clear completed lines and update score"""
        lines_to_clear = []
        for y in range(20):
            if all((x, y) in self.locked_positions for x in range(10)):
                lines_to_clear.append(y)
        
        if lines_to_clear:
            # Remove the lines
            for y in lines_to_clear:
                for x in range(10):
                    self.locked_positions.pop((x, y), None)
            
            # Move everything down
            new_positions = {}
            for (x, y), color in self.locked_positions.items():
                new_y = y
                for cleared_y in lines_to_clear:
                    if y < cleared_y:
                        new_y -= 1
                new_positions[(x, new_y)] = color
            self.locked_positions = new_positions
            
            # Update score
            self.lines_cleared += len(lines_to_clear)
            self.score += len(lines_to_clear) * 100 * self.level
            self.level = self.lines_cleared // 10 + 1

    @property
    def grid(self):
        """Get the current grid state"""
        grid = [[(0, 0, 0) for _ in range(10)] for _ in range(20)]
        
        # Add locked positions
        for (x, y), color in self.locked_positions.items():
            grid[y][x] = color
            
        # Add current piece
        if self.current_piece:
            for x, y in self.current_piece.get_positions():
                if 0 <= y < 20 and 0 <= x < 10:
                    grid[y][x] = self.current_piece.color
                    
        return grid 