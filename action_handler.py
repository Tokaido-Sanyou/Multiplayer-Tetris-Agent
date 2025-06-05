from utils import create_grid, hard_drop, get_shape_from_index
from piece_utils import valid_space
from constants import *

class ActionHandler:
    def __init__(self, player):
        self.player = player

    def move_left(self):
        self.player.current_piece.x -= 1
        if not (valid_space(self.player.current_piece, create_grid(self.player.locked_positions))):
            self.player.current_piece.x += 1

    def move_right(self):
        self.player.current_piece.x += 1
        if not (valid_space(self.player.current_piece, create_grid(self.player.locked_positions))):
            self.player.current_piece.x -= 1

    def move_down(self):
        self.player.current_piece.y += 1
        if not (valid_space(self.player.current_piece, create_grid(self.player.locked_positions))):
            self.player.current_piece.y -= 1

    def rotate_cw(self):
        self.player.current_piece.rotate(1, create_grid(self.player.locked_positions))

    def rotate_ccw(self):
        self.player.current_piece.rotate(-1, create_grid(self.player.locked_positions))

    def hard_drop(self):
        if hard_drop(self.player.current_piece, create_grid(self.player.locked_positions)):
            self.player.change_piece = True

    def hold_piece(self):
        if self.player.can_hold:
            if self.player.hold_piece is None:
                self.player.hold_piece = self.player.current_piece
                self.player.current_block_index += 1
                self.player.block_pool.ensure_blocks_ahead(self.player.current_block_index)
                self.player.current_piece = get_shape_from_index(self.player.block_pool.get_block_at(self.player.current_block_index))
                self.player.next_pieces = [get_shape_from_index(idx) for idx in 
                                        self.player.block_pool.get_next_blocks(self.player.current_block_index)]
            else:
                self.player.hold_piece, self.player.current_piece = self.player.current_piece, self.player.hold_piece
                self.player.current_piece.x, self.player.current_piece.y = 5, 0  # Reset position for held piece
            self.player.can_hold = False 