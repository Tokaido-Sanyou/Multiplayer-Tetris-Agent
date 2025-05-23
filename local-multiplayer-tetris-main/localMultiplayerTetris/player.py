import pygame
from .utils import get_shape_from_index, create_grid, clear_rows
from .piece_utils import convert_shape_format
from .action_handler import ActionHandler
from .key_handler import KeyHandler
from .block_pool import BlockPool
from .constants import *

class Player:
    def __init__(self, block_pool, is_player_one=True):
        self.locked_positions = {}
        self.current_piece = None
        self.next_pieces = []
        self.hold_piece = None
        self.can_hold = True
        self.score = 0
        self.change_piece = False
        self.is_player_one = is_player_one
        self.block_pool = block_pool
        self.current_block_index = 0
        
        # Initialize action and key handlers
        self.action_handler = ActionHandler(self)
        self.key_handler = KeyHandler(self.action_handler, is_player_one)
        
        # Initialize first piece and next pieces
        self.block_pool.ensure_blocks_ahead(self.current_block_index)
        first_piece_index = self.block_pool.get_block_at(self.current_block_index)
        self.current_piece = get_shape_from_index(first_piece_index)
        self.next_pieces = [get_shape_from_index(idx) for idx in 
                          self.block_pool.get_next_blocks(self.current_block_index)]

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            self.key_handler.handle_key(event.key)

    def update(self, fall_speed, level):
        if self.change_piece:
            shape_pos = convert_shape_format(self.current_piece)
            for pos in shape_pos:
                p = (pos[0], pos[1])
                self.locked_positions[p] = self.current_piece.color
            
            self.current_block_index += 1
            self.block_pool.ensure_blocks_ahead(self.current_block_index)
            self.current_piece = self.next_pieces[0]
            self.next_pieces = [get_shape_from_index(idx) for idx in 
                              self.block_pool.get_next_blocks(self.current_block_index)]
            self.change_piece = False
            self.can_hold = True
            
            # Handle line clearing and scoring
            grid = create_grid(self.locked_positions)
            lines_cleared = clear_rows(grid, self.locked_positions)
            self.score += lines_cleared * 10 * level
            
            return lines_cleared
        return 0 