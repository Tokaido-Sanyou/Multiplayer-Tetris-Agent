"""
Key handler for managing keyboard input
"""

import pygame

# Handle both direct execution and module import
try:
    from .action_handler import ActionHandler
except ImportError:
    # Direct execution - imports without relative paths
    from action_handler import ActionHandler

class KeyHandler:
    """Translates keyboard input to actions"""
    def __init__(self, action_handler, is_player_one=True):
        self.action_handler = action_handler
        self.is_player_one = is_player_one
        self.key_to_action = self._setup_key_mappings()

    def _setup_key_mappings(self):
        if self.is_player_one:
            return {
                pygame.K_a: self.action_handler.move_left,
                pygame.K_d: self.action_handler.move_right,
                pygame.K_s: self.action_handler.move_down,
                pygame.K_w: self.action_handler.rotate_cw,
                pygame.K_q: self.action_handler.rotate_ccw,
                pygame.K_SPACE: self.action_handler.hard_drop,
                pygame.K_c: self.action_handler.hold_piece
            }
        else:
            return {
                pygame.K_LEFT: self.action_handler.move_left,
                pygame.K_RIGHT: self.action_handler.move_right,
                pygame.K_DOWN: self.action_handler.move_down,
                pygame.K_UP: self.action_handler.rotate_cw,
                pygame.K_RSHIFT: self.action_handler.rotate_ccw,
                pygame.K_RETURN: self.action_handler.hard_drop,
                pygame.K_RCTRL: self.action_handler.hold_piece
            }

    def handle_key(self, key):
        """Handle a key press by executing the corresponding action"""
        if key in self.key_to_action:
            self.key_to_action[key]() 