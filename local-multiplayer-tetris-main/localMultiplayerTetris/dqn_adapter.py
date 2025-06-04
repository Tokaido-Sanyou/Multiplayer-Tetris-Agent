"""Adapter utilities to run the DQN agent from ``tetris-ai-master``
inside the modern ``TetrisEnv`` gym-style environment.

This file provides two public helpers:

1. ``board_props(grid_or_board) -> np.ndarray`` – replicate the four-element
   state encoding used by the original DQN implementation:  
   ``[lines_cleared, holes, total_bumpiness, sum_height]``.

2. ``enumerate_next_states(env)`` – given *the current* ``TetrisEnv``
   instance, brute-force every legal placement of the active piece and
   return a mapping ``{state_tuple: action}`` where *state_tuple* is the
   4-tuple produced by ``board_props`` and *action* is the corresponding
   placement index understood by ``TetrisEnv`` (0-39).

With these two helpers the pre-trained ``DQNAgent`` can be dropped in –
feed it the candidate state vectors, call ``best_state``, and translate the
result back to the discrete action.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import copy

# Local imports
from .utils import create_grid
from .piece_utils import convert_shape_format, valid_space
from .piece import Piece
from .constants import shapes as SHAPES

Board = np.ndarray  # alias for clarity (20x10 of 0/1 ints)


# ---------------------------------------------------------------------------
# 1. Feature extraction (mimics tetris-ai-master/Tetris._get_board_props)
# ---------------------------------------------------------------------------

def _count_completed_lines(board: Board) -> int:
    """Return how many *complete* lines (all 10 cells filled) exist."""
    return int(np.sum(board.sum(axis=1) == board.shape[1]))


def _count_holes(board: Board) -> int:
    """A *hole* is an empty cell with at least one block above it."""
    holes = 0
    for col in board.T:
        seen_block = False
        for cell in col:
            if cell:
                seen_block = True
            elif seen_block:
                holes += 1
    return holes


def _column_heights(board: Board) -> np.ndarray:
    """Height of each column measured from the bottom (range 0-20)."""
    heights = np.zeros(board.shape[1], dtype=int)
    for idx, col in enumerate(board.T):
        # first filled cell when scanning from top -> height
        filled = np.argmax(col)  # 0 if top is filled, else index of first 1
        if col[filled] == 0:  # column entirely empty
            heights[idx] = 0
        else:
            heights[idx] = board.shape[0] - filled
    return heights


def _bumpiness_and_sum_height(board: Board) -> Tuple[int, int]:
    heights = _column_heights(board)
    total_bumpiness = int(np.sum(np.abs(np.diff(heights))))
    sum_height = int(np.sum(heights))
    return total_bumpiness, sum_height


def board_props(grid_or_board) -> np.ndarray:
    """Return the four-element feature vector expected by the DQN.

    Accepts either the RGB *grid* (list[list[tuple]]) that
    ``localMultiplayerTetris.utils.create_grid`` returns or a plain 20x10
    integer *board* (values 0/1). The function normalises internally.
    """
    if isinstance(grid_or_board, np.ndarray):
        board = (grid_or_board > 0).astype(int)
    else:
        # assume list of lists of RGB tuples
        board = np.array([[1 if cell != (0, 0, 0) else 0 for cell in row]
                          for row in grid_or_board], dtype=int)

    lines = _count_completed_lines(board)
    holes = _count_holes(board)
    bumpiness, sum_height = _bumpiness_and_sum_height(board)
    return np.array([lines, holes, bumpiness, sum_height], dtype=float)


# ---------------------------------------------------------------------------
# 2. Brute-force enumeration of legal placements
# ---------------------------------------------------------------------------

def _piece_x_bounds(shape_positions):
    xs = [p[0] for p in shape_positions]
    return min(xs), max(xs)


def _place_piece_on_board(board: Board, positions) -> Board:
    new_board = board.copy()
    for x, y in positions:
        if 0 <= y < 20 and 0 <= x < 10:
            new_board[y, x] = 1
    return new_board


def enumerate_next_states(env) -> Dict[Tuple[float, float, float, float], int]:
    """Return mapping {state_tuple: action} for every *valid* placement.

    *env* must be an *active* TetrisEnv instance.
    """
    player = env.player
    grid_rgb = create_grid(player.locked_positions)
    base_board = np.array([[1 if cell != (0, 0, 0) else 0 for cell in row]
                           for row in grid_rgb], dtype=int)

    piece_shape = player.current_piece.shape

    next_states: Dict[Tuple[float, float, float, float], int] = {}

    # try all 4 orientation indices (0-3)
    for rot in range(4):
        temp_piece = Piece(0, 0, piece_shape)
        temp_piece.rotation = rot
        # horizontal bounds of this rotation so we know valid x range
        min_x, max_x = _piece_x_bounds(convert_shape_format(temp_piece))
        for col in range(-min_x, 10 - max_x):
            temp_piece.x = col
            temp_piece.y = 0
            # copy so dropping wont affect later iterations
            drop_piece = copy.deepcopy(temp_piece)

            # Simulate gravity – drop until collision
            while True:
                if valid_space(drop_piece, grid_rgb):
                    drop_piece.y += 1
                else:
                    drop_piece.y -= 1
                    break
            # If final y is negative the placement is above ceiling – skip
            if drop_piece.y < 0:
                continue

            placed_board = _place_piece_on_board(base_board, convert_shape_format(drop_piece))
            state_vec = tuple(board_props(placed_board))
            action = rot * 10 + col  # encoding used by TetrisEnv (<40)
            next_states[state_vec] = action

    return next_states
