#!/usr/bin/env python3
"""
Enhanced Multiplayer Tetris with Mode Selection

Features:
- Mode Selection: Normal Mode vs Locked Position Mode
- Locked Mode: Shows valid positions with rotation-first navigation
- Normal Mode: Standard Tetris gameplay
- Fixed navigation: Arrows cycle rotations first, then positions
- Immediate placement visibility when pieces spawn
- No falling blocks in locked mode
"""

import pygame
import sys
import os
import time
import copy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.game.game import Game
from envs.game.constants import s_width, s_height, top_left_x, top_left_y, block_size, mid_x, shapes
from envs.game.utils import create_grid, hard_drop
from envs.game.piece_utils import valid_space, convert_shape_format
from envs.game.action_handler import ActionHandler
from envs.game.piece import Piece


class GameModeSelector:
    """Mode selection screen for choosing between Normal and Locked Position modes"""
    
    def __init__(self, surface):
        self.surface = surface
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.selected_mode = 0  # 0 = Normal, 1 = Locked Position
        self.modes = ["Normal Mode", "Locked Position Mode"]
        
    def handle_input(self, event):
        """Handle mode selection input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                self.selected_mode = 1 - self.selected_mode  # Toggle between 0 and 1
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                return self.selected_mode  # Return selected mode
            elif event.key == pygame.K_ESCAPE:
                return -1  # Exit
        return None
    
    def draw(self):
        """Draw mode selection screen"""
        self.surface.fill((0, 0, 0))
        
        # Title
        title_text = self.font_large.render("Select Game Mode", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(s_width // 2, s_height // 4))
        self.surface.blit(title_text, title_rect)
        
        # Mode options
        for i, mode in enumerate(self.modes):
            color = (255, 255, 0) if i == self.selected_mode else (255, 255, 255)
            mode_text = self.font_medium.render(f"{'> ' if i == self.selected_mode else '  '}{mode}", True, color)
            mode_rect = mode_text.get_rect(center=(s_width // 2, s_height // 2 + i * 50))
            self.surface.blit(mode_text, mode_rect)
        
        # Descriptions
        descriptions = [
            "Standard multiplayer Tetris gameplay",
            "Show all valid positions with rotation-first navigation"
        ]
        
        desc_text = self.font_small.render(descriptions[self.selected_mode], True, (200, 200, 200))
        desc_rect = desc_text.get_rect(center=(s_width // 2, s_height // 2 + 150))
        self.surface.blit(desc_text, desc_rect)
        
        # Instructions
        inst_text = self.font_small.render("Use UP/DOWN to select, ENTER/SPACE to confirm, ESC to exit", True, (150, 150, 150))
        inst_rect = inst_text.get_rect(center=(s_width // 2, s_height - 50))
        self.surface.blit(inst_text, inst_rect)
        
        pygame.display.update()


class NormalMultiplayerGame:
    """Standard multiplayer Tetris game"""
    
    def __init__(self, surface):
        self.surface = surface
        self.game = Game(surface, auto_start=True)
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        print("🎮 Normal Multiplayer Game Initialized")
        print("Standard Tetris controls for both players")
        print("ESC to return to mode selection, P to pause")
    
    def handle_input(self, event):
        """Handle normal game input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
                return False
            elif event.key == pygame.K_p:
                self.paused = not self.paused
                return True
        
        # Pass through to game's input handler
        if not self.paused:
            self.game.handle_input(event)
        
        return True
    
    def update(self):
        """Update normal game"""
        if not self.paused:
            return self.game.update()
        return True
    
    def draw(self):
        """Draw normal game"""
        self.game.draw()
        
        if self.paused:
            self.draw_pause_screen()
        
        pygame.display.update()
    
    def draw_pause_screen(self):
        """Draw pause overlay"""
        overlay = pygame.Surface((s_width, s_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.surface.blit(overlay, (0, 0))
        
        font = pygame.font.Font(None, 48)
        pause_text = font.render("PAUSED", True, (255, 255, 255))
        text_rect = pause_text.get_rect(center=(s_width // 2, s_height // 2))
        self.surface.blit(pause_text, text_rect)
    
    def run(self):
        """Main game loop for normal mode"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    if not self.handle_input(event):
                        break
            
            if not self.update():
                self.show_game_over()
                break
            
            self.draw()
            self.clock.tick(60)
        
        return "menu"  # Return to mode selection
    
    def show_game_over(self):
        """Show game over screen"""
        overlay = pygame.Surface((s_width, s_height))
        overlay.set_alpha(192)
        overlay.fill((0, 0, 0))
        self.surface.blit(overlay, (0, 0))
        
        font = pygame.font.Font(None, 48)
        game_over_text = font.render("GAME OVER", True, (255, 0, 0))
        text_rect = game_over_text.get_rect(center=(s_width // 2, s_height // 2 - 50))
        self.surface.blit(game_over_text, text_rect)
        
        font_small = pygame.font.Font(None, 24)
        restart_text = font_small.render("Press ESC to return to mode selection", True, (255, 255, 255))
        restart_rect = restart_text.get_rect(center=(s_width // 2, s_height // 2 + 50))
        self.surface.blit(restart_text, restart_rect)
        
        pygame.display.update()
        
        # Wait for input
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    self.running = False
                elif event.key == pygame.K_ESCAPE:
                    waiting = False
                    self.running = False


class LockedPositionGame:
    """Enhanced locked position mode with simplified position-only navigation and timed lock-in"""
    
    def __init__(self, surface):
        self.surface = surface
        self.game = Game(surface, auto_start=False)
        self.clock = pygame.time.Clock()
        
        # Simplified navigation system - position-only (no rotation control)
        self.valid_placements = [[], []]  # Per-player valid placements
        self.current_position = [0, 0]  # Current position index per player
        
        # Timing system for lock-in
        self.lock_timeout = 10.0  # 10 seconds per piece
        self.lock_timers = [self.lock_timeout, self.lock_timeout]  # Per-player timers
        self.timer_start_time = [None, None]  # Track when timer started
        
        # Block display control
        self.blocks_visible = [True, True]  # Control visibility per player
        self.last_pieces = [None, None]  # Track piece changes
        
        # Game state
        self.paused = False
        self.running = True
        
        # Start the game to initialize players properly
        self.game.start()
        
        # Initialize valid placements and start timers
        self.update_valid_placements()
        self.start_lock_timers()
        
        print("🎮 Locked Position Mode Initialized")
        print("SIMPLIFIED: Position-only navigation with 10-second lock timeout")
        print("Player 1: A/D navigate positions, SPACE to place")
        print("Player 2: Left/Right navigate positions, ENTER to place") 
        print("ESC to return to mode selection, P to pause")
    
    def start_lock_timers(self):
        """Start lock timers for both players"""
        import time
        current_time = time.time()
        for player_idx in range(2):
            if self.valid_placements[player_idx]:  # Only start timer if there are valid positions
                self.timer_start_time[player_idx] = current_time
                self.lock_timers[player_idx] = self.lock_timeout
    
    def update_lock_timers(self):
        """Update lock timers and auto-place on timeout"""
        if self.paused:
            return
        
        import time
        current_time = time.time()
        
        for player_idx in range(2):
            if self.timer_start_time[player_idx] is not None:
                elapsed = current_time - self.timer_start_time[player_idx]
                self.lock_timers[player_idx] = max(0, self.lock_timeout - elapsed)
                
                # Auto-place on timeout
                if self.lock_timers[player_idx] <= 0:
                    print(f"⏰ Player {player_idx + 1} timeout - auto-placing piece")
                    self.place_selected_piece(player_idx)

    def update_valid_placements(self):
        """Find ALL valid locked positions with navigation state preservation"""
        for player_idx in range(2):
            player = self.game.player1 if player_idx == 0 else self.game.player2
            if not player.current_piece:
                continue
                
            # Store current piece info for comparison
            current_piece_info = (
                tuple(player.current_piece.shape) if player.current_piece.shape else None,
                player.current_piece.x,
                player.current_piece.y
            )
            
            # Find all valid placements including wall kicks and tucked positions
            new_placements = self.find_best_placements_per_position(player)
            
            # Preserve navigation state if piece hasn't changed
            old_placements = self.valid_placements[player_idx]
            if (hasattr(self, 'last_piece_info') and 
                hasattr(self.last_piece_info, '__getitem__') and
                len(self.last_piece_info) > player_idx and
                self.last_piece_info[player_idx] == current_piece_info and
                old_placements and new_placements):
                # Same piece - preserve current position if still valid
                current_pos = self.current_position[player_idx]
                if current_pos < len(new_placements):
                    # Keep current position
                    pass
                else:
                    # Reset to first position if current is out of bounds
                    self.current_position[player_idx] = 0
            else:
                # New piece or first time - reset navigation to first position
                self.current_position[player_idx] = 0
                
                # Store piece info for next comparison
                if not hasattr(self, 'last_piece_info'):
                    self.last_piece_info = [None, None]
                self.last_piece_info[player_idx] = current_piece_info
            
            self.valid_placements[player_idx] = new_placements
            
            # Show blocks for this player
            self.blocks_visible[player_idx] = True

    def find_advanced_tuck_positions(self, player):
        """Find advanced tuck positions that require multiple moves"""
        if not player.current_piece:
            return []
        
        from envs.game.utils import create_grid
        from envs.game.piece_utils import valid_space, convert_shape_format
        
        grid = create_grid(player.locked_positions)
        advanced_tucks = []
        
        # Test sliding from different starting positions
        for rotation in range(len(player.current_piece.shape)):
            for start_x in range(-3, 13):  # Allow starting outside grid
                for slide_direction in [-1, 1]:  # Left and right sliding
                    
                    # Create test piece at starting position
                    test_piece = type(player.current_piece)(start_x, 0, player.current_piece.shape)
                    test_piece.rotation = rotation
                    
                    # Drop to find landing position
                    while test_piece.y < 25:
                        if not valid_space(test_piece, grid):
                            test_piece.y -= 1
                            break
                        test_piece.y += 1
                    
                    if not valid_space(test_piece, grid):
                        continue
                    
                    # Try sliding in the given direction
                    max_slides = 8  # Maximum slide distance
                    for slide_steps in range(1, max_slides + 1):
                        slide_x = start_x + (slide_direction * slide_steps)
                        
                        # Create slid piece
                        slid_piece = type(player.current_piece)(slide_x, test_piece.y, player.current_piece.shape)
                        slid_piece.rotation = rotation
                        
                        if valid_space(slid_piece, grid):
                            # Check if this creates a tuck (piece fits under overhang)
                            piece_positions = convert_shape_format(slid_piece)
                            
                            is_advanced_tuck = False
                            for pos in piece_positions:
                                px, py = pos
                                
                                # Check for overhang above this position
                                if py > 0:
                                    overhang_found = False
                                    for check_y in range(max(0, py - 3), py):
                                        # Check for blocks creating overhang
                                        if ((px - 1, check_y) in grid or (px + 1, check_y) in grid) and (px, check_y) not in grid:
                                            overhang_found = True
                                            break
                                    
                                    if overhang_found:
                                        is_advanced_tuck = True
                                        break
                                
                                # Check for tight horizontal spaces (blocks on both sides)
                                if (px - 1, py) in grid and (px + 1, py) in grid:
                                    is_advanced_tuck = True
                                    break
                                
                                # Check for deep positions (piece is significantly below surface)
                                surface_height = 0
                                for check_y in range(20):
                                    if any((px + dx, check_y) in grid for dx in range(-2, 3) if 0 <= px + dx < 10):
                                        surface_height = check_y
                                        break
                                
                                if py > surface_height + 2:  # Piece is 2+ rows below surface
                                    is_advanced_tuck = True
                                    break
                            
                            if is_advanced_tuck:
                                # Ensure positions are within grid bounds
                                valid_placement = True
                                for pos in piece_positions:
                                    px, py = pos
                                    if px < 0 or px >= 10 or py < 0 or py >= 20:
                                        valid_placement = False
                                        break
                                
                                if valid_placement:
                                    advanced_tucks.append({
                                        'x': slide_x,
                                        'y': slid_piece.y,
                                        'rotation': rotation,
                                        'positions': [(pos[0], pos[1]) for pos in piece_positions],
                                        'start_x': start_x,
                                        'slide_direction': slide_direction,
                                        'slide_steps': slide_steps,
                                        'is_advanced_tuck': True
                                    })
                        else:
                            break  # Can't slide further in this direction
        
        return advanced_tucks

    def find_best_placements_per_position(self, player):
        """Find ALL valid placements including tucked positions with wall kicks"""
        if not player.current_piece:
            return []
        
        valid_placements = []
        grid = create_grid(player.locked_positions)
        
        # Import wall kick data for advanced placement detection
        from envs.game.game import JLSTZ_WALL_KICKS, I_WALL_KICKS
        
        # Test each x position and ALL rotations with wall kicks
        for x in range(10):  # Full grid width
            for rotation in range(len(player.current_piece.shape)):
                # Try basic placement first
                placements = self._try_placement_with_kicks(
                    player.current_piece, x, rotation, grid, 
                    JLSTZ_WALL_KICKS, I_WALL_KICKS
                )
                valid_placements.extend(placements)
        
        # Add advanced tuck positions
        advanced_tucks = self.find_advanced_tuck_positions(player)
        valid_placements.extend(advanced_tucks)
        
        # Remove duplicates (same position and rotation)
        unique_placements = []
        seen = set()
        for placement in valid_placements:
            key = (placement['x'], placement['y'], placement['rotation'])
            if key not in seen:
                seen.add(key)
                unique_placements.append(placement)
        
        return unique_placements
    
    def _try_placement_with_kicks(self, piece, x, rotation, grid, jlstz_kicks, i_kicks):
        """Try placement with wall kicks for tucked positions"""
        placements = []
        
        # Basic placement attempt (no kicks)
        test_piece = copy.deepcopy(piece)
        test_piece.x = x
        test_piece.rotation = rotation
        test_piece.y = 0
        
        if valid_space(test_piece, grid):
            hard_drop(test_piece, grid)
            if valid_space(test_piece, grid):
                final_positions = convert_shape_format(test_piece)
                if final_positions and len(final_positions) == 4:
                    # Check if all positions are within bounds (prevent out-of-bounds placements)
                    valid_bounds = all(0 <= pos[0] < 10 and 0 <= pos[1] < 20 for pos in final_positions)
                    if valid_bounds:
                        # Check for floating pieces (pieces with no support below)
                        is_floating = False
                        piece_positions_set = set((pos[0], pos[1]) for pos in final_positions)
                        for pos in final_positions:
                            px, py = pos
                            if py < 19:  # Not on bottom row
                                # Check if there's support below this position
                                has_support = False
                                for check_y in range(py + 1, 20):
                                    # Check locked positions or other piece positions (grid[y][x] format)
                                    if ((px, check_y) in piece_positions_set or 
                                        (0 <= px < 10 and 0 <= check_y < 20 and grid[check_y][px] != (0, 0, 0))):
                                        has_support = True
                                        break
                                if not has_support:
                                    is_floating = True
                                    break
                        
                        # Only add placement if it's not floating
                        if not is_floating:
                            placements.append({
                                'x': test_piece.x,
                                'y': test_piece.y,
                                'rotation': rotation,
                                'positions': final_positions,
                                'piece_type': piece.shape
                            })
        
        # Try wall kicks for tucked positions
        # Determine which wall kick data to use
        piece_type = piece.shape
        is_i_piece = (len(piece_type) == 2 and 
                     '0000' in str(piece_type[0]) if piece_type else False)
        
        kick_data = i_kicks if is_i_piece else jlstz_kicks
        
        # Try kicks from current rotation to target rotation
        if rotation in kick_data:
            for target_rot in kick_data[rotation]:
                for kick_x, kick_y in kick_data[rotation][target_rot]:
                    kicked_piece = copy.deepcopy(piece)
                    kicked_piece.x = x + kick_x
                    kicked_piece.rotation = target_rot
                    kicked_piece.y = kick_y
                    
                    # Ensure kicked position is within bounds
                    if 0 <= kicked_piece.x < 10:
                        if valid_space(kicked_piece, grid):
                            # Hard drop from kicked position
                            hard_drop(kicked_piece, grid)
                            if valid_space(kicked_piece, grid):
                                final_positions = convert_shape_format(kicked_piece)
                                if final_positions and len(final_positions) == 4:
                                    # Check if all positions are within bounds (prevent out-of-bounds placements)
                                    valid_bounds = all(0 <= pos[0] < 10 and 0 <= pos[1] < 20 for pos in final_positions)
                                    if not valid_bounds:
                                        continue  # Skip this invalid placement
                                    
                                    # Check for floating pieces in wall kick placements too
                                    is_floating = False
                                    piece_positions_set = set((pos[0], pos[1]) for pos in final_positions)
                                    for pos in final_positions:
                                        px, py = pos
                                        if py < 19:  # Not on bottom row
                                            # Check if there's support below this position
                                            has_support = False
                                            for check_y in range(py + 1, 20):
                                                # Check locked positions or other piece positions (grid[y][x] format)
                                                if ((px, check_y) in piece_positions_set or
                                                    (0 <= px < 10 and 0 <= check_y < 20 and grid[check_y][px] != (0, 0, 0))):
                                                    has_support = True
                                                    break
                                            if not has_support:
                                                is_floating = True
                                                break
                                    
                                    # Only add wall kick placement if it's not floating
                                    if not is_floating:
                                        placements.append({
                                            'x': kicked_piece.x,
                                            'y': kicked_piece.y,
                                            'rotation': target_rot,
                                            'positions': final_positions,
                                            'piece_type': piece.shape
                                        })
        
        return placements

    def handle_input(self, event):
        """Handle input with simplified position-only navigation"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
                return False
            elif event.key == pygame.K_p:
                self.paused = not self.paused
                return True
            
            if not self.paused:
                # Player 1 controls - SIMPLIFIED: Only A/D for position navigation
                if event.key == pygame.K_a:  # Previous position
                    self.navigate_position(0, -1)
                elif event.key == pygame.K_d:  # Next position
                    self.navigate_position(0, 1)
                elif event.key == pygame.K_SPACE:  # Place selected piece
                    self.place_selected_piece(0)
                
                # Player 2 controls - SIMPLIFIED: Only Left/Right for position navigation
                elif event.key == pygame.K_LEFT:  # Previous position
                    self.navigate_position(1, -1)
                elif event.key == pygame.K_RIGHT:  # Next position
                    self.navigate_position(1, 1)
                elif event.key == pygame.K_RETURN:  # Place selected piece
                    self.place_selected_piece(1)
        
        return True

    def navigate_position(self, player_idx, direction):
        """Navigate through positions"""
        if not self.valid_placements[player_idx]:
            return
        
        num_positions = len(self.valid_placements[player_idx])
        if num_positions > 0:
            self.current_position[player_idx] = (self.current_position[player_idx] + direction) % num_positions

    def get_current_selection(self, player_idx):
        """Get currently selected placement"""
        if not self.valid_placements[player_idx]:
            return None
        
        if 0 <= self.current_position[player_idx] < len(self.valid_placements[player_idx]):
            return self.valid_placements[player_idx][self.current_position[player_idx]]
        
        return None

    def output_position_series(self, player_idx):
        """Output position series for current selection"""
        placement = self.get_current_selection(player_idx)
        if placement:
            print(f"Player {player_idx + 1} - Rotation: {placement['rotation']}, Position: ({placement['x']}, {placement['y']})")

    def place_selected_piece(self, player_idx):
        """Place the currently selected piece and output its series"""
        placement = self.get_current_selection(player_idx)
        if not placement:
            return
        
        # Output the placement series
        self.output_position_series(player_idx)
        
        # Hide blocks for this player after placement
        self.blocks_visible[player_idx] = False
        
        # Apply the placement to the game
        player = self.game.player1 if player_idx == 0 else self.game.player2
        if player.current_piece:
            # Set piece to selected position and rotation
            player.current_piece.x = placement['x']
            player.current_piece.y = placement['y']
            player.current_piece.rotation = placement['rotation']
            
            # Lock the piece in place (from game logic)
            piece_pos = convert_shape_format(player.current_piece)
            for pos in piece_pos:
                x, y = pos
                if y > -1:
                    player.locked_positions[(x, y)] = player.current_piece.color
            
            # Clear completed lines using the game's clear_rows function
            from envs.game.utils import clear_rows, create_grid
            grid = create_grid(player.locked_positions)
            lines_cleared = clear_rows(grid, player.locked_positions)
            player.score += lines_cleared * 10 * self.game.level
            
            # Send garbage lines to opponent if lines were cleared
            if lines_cleared > 0:
                self.send_garbage_lines(player_idx, lines_cleared)
            
            # Get next piece using the player's block pool system
            player.current_block_index += 1
            player.block_pool.ensure_blocks_ahead(player.current_block_index)
            player.current_piece = player.next_pieces[0]
            
            # Update next pieces list
            from envs.game.utils import get_shape_from_index
            player.next_pieces = [get_shape_from_index(idx) for idx in 
                                player.block_pool.get_next_blocks(player.current_block_index)]
            player.can_hold = True
            
            # Update game grids to reflect new locked positions
            self.game.p1_grid = create_grid(self.game.player1.locked_positions)
            self.game.p2_grid = create_grid(self.game.player2.locked_positions)
            
            # Immediately update valid placements for new piece and restart timer
            self.update_valid_placements()
            import time
            self.timer_start_time[player_idx] = time.time()
            self.blocks_visible[player_idx] = True  # Show blocks for new piece
    
    def update(self):
        """Update with no falling blocks in locked mode, including lock timers and game over detection"""
        if not self.paused:
            # Check for game over conditions first
            if self.check_game_over():
                return False  # Signal game over
            
            # Update lock timers (handles auto-placement on timeout)  
            self.update_lock_timers()
            
            # Check for piece changes and update immediately
            current_pieces = [
                (self.game.player1.current_piece.shape, self.game.player1.current_piece.x, self.game.player1.current_piece.y) if self.game.player1.current_piece else None,
                (self.game.player2.current_piece.shape, self.game.player2.current_piece.x, self.game.player2.current_piece.y) if self.game.player2.current_piece else None
            ]
            
            # Update placements when pieces change
            if current_pieces != self.last_pieces:
                self.update_valid_placements()
                self.last_pieces = current_pieces
            
            # Update only non-falling game components (timers, etc.)
            # Skip piece falling logic
            return True
        
        return True
    
    def check_game_over(self):
        """Check if either player has filled their board (game over condition)"""
        # Don't check game over during initialization or if game hasn't started
        if not hasattr(self.game, 'player1') or not hasattr(self.game, 'player2'):
            return False
        
        for player_idx in range(2):
            player = self.game.player1 if player_idx == 0 else self.game.player2
            
            # Skip check if we don't have a current piece (during initialization)
            if not player.current_piece:
                continue
            
            # Check if top rows are filled first (classic Tetris game over)
            grid = create_grid(player.locked_positions)
            # Check for non-empty cells (not (0,0,0)) in top rows
            top_row_filled = any(cell != (0, 0, 0) for cell in grid[0])
            second_row_filled = any(cell != (0, 0, 0) for cell in grid[1])
            if top_row_filled or second_row_filled:
                print(f"🎮 Game Over: Player {player_idx + 1} board filled to top!")
                self.show_game_over(player_idx)
                return True
            
            # Only check valid placements if they have been properly initialized AND game has started
            if (hasattr(self, 'valid_placements') and 
                len(self.valid_placements) > player_idx and
                hasattr(self, 'timer_start_time') and 
                self.timer_start_time[player_idx] is not None):  # Game has started
                
                # Check if no valid placements exist (board too full)
                if len(self.valid_placements[player_idx]) == 0:
                    print(f"🎮 Game Over: Player {player_idx + 1} has no valid placements!")
                    self.show_game_over(player_idx)
                    return True
        
        return False
    
    def show_game_over(self, losing_player_idx):
        """Show game over screen"""
        winner = 2 - losing_player_idx  # Other player wins
        print(f"🏆 Player {winner} Wins!")
        
        # Display game over message
        font = pygame.font.Font(None, 48)
        game_over_text = font.render(f"GAME OVER - Player {winner} Wins!", True, (255, 255, 255))
        restart_text = font.render("Press R to restart or ESC to exit", True, (200, 200, 200))
        
        # Center text on screen
        text_rect = game_over_text.get_rect(center=(s_width // 2, s_height // 2))
        restart_rect = restart_text.get_rect(center=(s_width // 2, s_height // 2 + 60))
        
        # Draw semi-transparent overlay
        overlay = pygame.Surface((s_width, s_height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.surface.blit(overlay, (0, 0))
        
        # Draw text
        self.surface.blit(game_over_text, text_rect)
        self.surface.blit(restart_text, restart_rect)
        
        pygame.display.update()
        
        # Wait for restart or exit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Restart game
                        self.__init__(self.surface)
                        return True
                    elif event.key == pygame.K_ESCAPE:
                        return False
        
        return True
    
    def send_garbage_lines(self, sending_player_idx, lines_cleared):
        """Send garbage lines to opponent when clearing lines"""
        if lines_cleared <= 0:
            return
        
        # Determine receiving player
        receiving_player_idx = 1 - sending_player_idx
        receiving_player = self.game.player1 if receiving_player_idx == 0 else self.game.player2
        
        # Calculate garbage lines to send (standard Tetris rules)
        garbage_to_send = 0
        if lines_cleared == 1:
            garbage_to_send = 0  # Single line clears don't send garbage
        elif lines_cleared == 2:
            garbage_to_send = 1
        elif lines_cleared == 3:
            garbage_to_send = 2
        elif lines_cleared == 4:  # Tetris
            garbage_to_send = 4
        
        if garbage_to_send > 0:
            print(f"🗑️ Sending {garbage_to_send} garbage lines from P{sending_player_idx + 1} to P{receiving_player_idx + 1}")
            
            # Add garbage lines to receiving player's board
            self.add_garbage_lines(receiving_player_idx, garbage_to_send)
    
    def add_garbage_lines(self, player_idx, num_lines):
        """Add garbage lines to a player's board"""
        player = self.game.player1 if player_idx == 0 else self.game.player2
        
        # Shift existing blocks up
        new_positions = {}
        for (x, y), color in player.locked_positions.items():
            new_y = y - num_lines
            if new_y >= 0:  # Keep blocks that fit on screen
                new_positions[(x, new_y)] = color
        
        # Add garbage lines at bottom
        import random
        for line in range(num_lines):
            y_pos = 19 - line  # Bottom lines
            gap_x = random.randint(0, 9)  # Random gap position
            
            for x in range(10):
                if x != gap_x:  # Leave one gap per line
                    new_positions[(x, y_pos)] = (128, 128, 128)  # Gray garbage color
        
        # Update player's locked positions
        player.locked_positions = new_positions
        
        # Update the grid
        if player_idx == 0:
            self.game.p1_grid = create_grid(player.locked_positions)
        else:
            self.game.p2_grid = create_grid(player.locked_positions)

    def draw(self):
        """Draw the locked position game with controlled block visibility"""
        # Draw base game without the falling pieces
        self.surface.fill((33,29,29))
        
        # Import draw functions
        from envs.game.utils import draw_window, draw_next_pieces, draw_hold_piece
        from envs.game.constants import mid_x
        
        # Update grids with current locked positions before drawing
        self.game.p1_grid = create_grid(self.game.player1.locked_positions)
        self.game.p2_grid = create_grid(self.game.player2.locked_positions)
        
        # Draw window without current pieces (pass None for current pieces)
        draw_window(self.surface, self.game.p1_grid, self.game.p2_grid,
                   None, None,  # No falling pieces in locked mode
                   self.game.player1.score, self.game.player2.score,
                   self.game.level, self.game.fall_speed, add=int(mid_x))
        
        # Draw next pieces and hold pieces
        draw_next_pieces(self.game.player1.next_pieces, self.surface, 0)
        draw_next_pieces(self.game.player2.next_pieces, self.surface, 1)
        draw_hold_piece(self.game.player1.hold_piece, self.surface, 0)
        draw_hold_piece(self.game.player2.hold_piece, self.surface, 1)
        
        # Draw valid placements only if blocks are visible for that player
        if not self.paused:
            if self.blocks_visible[0]:
                self.draw_all_valid_placements(0, (255, 0, 255))  # Player 1 - Bright Magenta
            if self.blocks_visible[1]:
                self.draw_all_valid_placements(1, (0, 255, 255))  # Player 2 - Bright Cyan
        
        # Draw UI information
        self.draw_ui()
        
        # Draw pause screen if paused
        if self.paused:
            self.draw_pause_screen()
        
        pygame.display.update()

    def draw_all_valid_placements(self, player_idx, color):
        """Draw all valid placements with current selection highlighted"""
        if not self.valid_placements[player_idx]:
            return
        
        # Calculate screen offset for player 2
        from envs.game.constants import mid_x
        offset_x = int(mid_x) if player_idx == 1 else 0
        
        current_selection = self.get_current_selection(player_idx)
        
        for placement in self.valid_placements[player_idx]:
            # Check if this is the currently selected placement
            is_selected = (current_selection and 
                          placement['x'] == current_selection['x'] and 
                          placement['y'] == current_selection['y'] and 
                          placement['rotation'] == current_selection['rotation'])
            
            if is_selected:
                # Selected placement: bright white with high opacity
                block_color = (255, 255, 255)
                alpha = 220
                outline_width = 3
            else:
                # Other valid placements: player color with lower opacity
                block_color = color
                alpha = 60
                outline_width = 1
            
            # Draw each block position in this placement
            for pos in placement['positions']:
                x, y = pos
                if 0 <= x < 10 and 0 <= y < 20:  # Within grid bounds
                    from envs.game.constants import top_left_x, top_left_y, block_size
                    block_x = int(top_left_x) + offset_x + x * block_size
                    block_y = int(top_left_y) + y * block_size
                    
                    # Create semi-transparent surface
                    block_surface = pygame.Surface((block_size, block_size))
                    block_surface.set_alpha(alpha)
                    block_surface.fill(block_color)
                    
                    # Draw the transparent block
                    self.surface.blit(block_surface, (block_x, block_y))
                    
                    # Draw outline for visibility
                    pygame.draw.rect(self.surface, block_color, 
                                   (block_x, block_y, block_size, block_size), outline_width)

    def draw_ui(self):
        """Draw enhanced UI information with timer display"""
        font = pygame.font.Font(None, 24)
        
        # Player 1 information with timer
        p1_placements = len(self.valid_placements[0])
        current_p1 = self.get_current_selection(0)
        timer_p1 = self.lock_timers[0]
        if current_p1:
            p1_text = f"P1: Pos {self.current_position[0]+1}/{p1_placements}, Rot {current_p1['rotation']} | Timer: {timer_p1:.1f}s"
        else:
            p1_text = f"P1: No valid placements"
        p1_surface = font.render(p1_text, True, (255, 0, 255))
        self.surface.blit(p1_surface, (10, 10))
        
        # Player 2 information with timer
        p2_placements = len(self.valid_placements[1])
        current_p2 = self.get_current_selection(1)
        timer_p2 = self.lock_timers[1]
        if current_p2:
            p2_text = f"P2: Pos {self.current_position[1]+1}/{p2_placements}, Rot {current_p2['rotation']} | Timer: {timer_p2:.1f}s"
        else:
            p2_text = f"P2: No valid placements"
        p2_surface = font.render(p2_text, True, (0, 255, 255))
        self.surface.blit(p2_surface, (10, 40))
        
        # Status information
        status_text = "LOCKED MODE: Position-only navigation with 10s timeout"
        status_surface = font.render(status_text, True, (255, 255, 255))
        self.surface.blit(status_surface, (10, 70))
        
        # Controls - Updated
        controls_text = "P1: A/D=Navigate Positions, Space=Place | P2: ←/→=Navigate Positions, Enter=Place"
        controls_surface = font.render(controls_text, True, (255, 255, 255))
        from envs.game.constants import s_height
        self.surface.blit(controls_surface, (10, s_height - 30))
    
    def draw_pause_screen(self):
        """Draw pause overlay"""
        overlay = pygame.Surface((s_width, s_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.surface.blit(overlay, (0, 0))
        
        font = pygame.font.Font(None, 48)
        pause_text = font.render("PAUSED", True, (255, 255, 255))
        text_rect = pause_text.get_rect(center=(s_width // 2, s_height // 2))
        self.surface.blit(pause_text, text_rect)
    
    def run(self):
        """Main game loop for locked position mode"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    if not self.handle_input(event):
                        break
            
            if not self.update():
                break
            
            self.draw()
            self.clock.tick(60)
        
        return "menu"  # Return to mode selection


def main():
    """Main function with mode selection"""
    pygame.init()
    surface = pygame.display.set_mode((s_width, s_height))
    pygame.display.set_caption("Enhanced Multiplayer Tetris - Mode Selection")
    
    print("✓ Hardware acceleration enabled")
    print("🎮 Enhanced Multiplayer Tetris with Mode Selection")
    print("==================================================")
    
    clock = pygame.time.Clock()
    
    while True:
        # Mode selection
        mode_selector = GameModeSelector(surface)
        selected_mode = None
        
        while selected_mode is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                else:
                    selected_mode = mode_selector.handle_input(event)
                    if selected_mode == -1:  # Exit
                        pygame.quit()
                        sys.exit()
            
            mode_selector.draw()
            clock.tick(60)
        
        # Run selected game mode
        if selected_mode == 0:  # Normal Mode
            game = NormalMultiplayerGame(surface)
            result = game.run()
        elif selected_mode == 1:  # Locked Position Mode
            game = LockedPositionGame(surface)
            result = game.run()
        
        # Check if we should return to menu or exit
        if result != "menu":
            break
    
    print("👋 Thanks for playing!")
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main() 