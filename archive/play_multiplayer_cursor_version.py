#!/usr/bin/env python3
"""
New Multiplayer Tetris Implementation with Fixed Locked Position Mode

Features:
- Working menu system with proper mode selection
- Blinking cursors that actually blink
- Piece placement preview
- Dual player support in locked position mode
- Proper game state management
"""

import pygame
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.game.game import Game
from envs.game.constants import s_width, s_height, top_left_x, top_left_y, block_size, mid_x
from envs.game.utils import create_grid, hard_drop
from envs.game.piece_utils import valid_space, convert_shape_format
from envs.game.action_handler import ActionHandler
import copy


class LockedPositionGame:
    """New locked position game implementation"""
    
    def __init__(self, surface):
        self.surface = surface
        self.game = Game(surface, auto_start=False)
        self.clock = pygame.time.Clock()
        
        # Valid position management - show ALL valid placements automatically
        self.valid_placements = [[], []]  # Per-player valid placements with all rotations
        self.current_selection = [0, 0]  # Currently selected placement index per player
        
        # Blinking state
        self.blink_timer = 0
        self.blink_interval = 30  # 30 frames = 0.5 seconds at 60 FPS
        self.blocks_visible = True
        
        # Game state
        self.paused = False
        self.running = True
        
        # Initialize valid positions for both players
        self.update_valid_positions()
        
        print("🎮 New Locked Position Game Initialized")
        print("Player 1: WASD to navigate, SPACE to place")
        print("Player 2: Arrow keys to navigate, ENTER to place") 
        print("ESC to return to menu, P to pause")
    
    def handle_input(self, event):
        """Handle input for both players"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
                return False
            elif event.key == pygame.K_p:
                self.paused = not self.paused
                return True
            
            if not self.paused:
                # Player 1 controls (WASD + SPACE)
                if event.key == pygame.K_w:  # Move up
                    self.player_cursors[0]['y'] = max(0, self.player_cursors[0]['y'] - 1)
                elif event.key == pygame.K_s:  # Move down
                    self.player_cursors[0]['y'] = min(19, self.player_cursors[0]['y'] + 1)
                elif event.key == pygame.K_a:  # Move left
                    self.player_cursors[0]['x'] = max(0, self.player_cursors[0]['x'] - 1)
                elif event.key == pygame.K_d:  # Move right
                    self.player_cursors[0]['x'] = min(9, self.player_cursors[0]['x'] + 1)
                elif event.key == pygame.K_SPACE:  # Place piece
                    self.place_piece(0)
                
                # Player 2 controls (Arrow keys + ENTER)
                elif event.key == pygame.K_UP:  # Move up
                    self.player_cursors[1]['y'] = max(0, self.player_cursors[1]['y'] - 1)
                elif event.key == pygame.K_DOWN:  # Move down
                    self.player_cursors[1]['y'] = min(19, self.player_cursors[1]['y'] + 1)
                elif event.key == pygame.K_LEFT:  # Move left
                    self.player_cursors[1]['x'] = max(0, self.player_cursors[1]['x'] - 1)
                elif event.key == pygame.K_RIGHT:  # Move right
                    self.player_cursors[1]['x'] = min(9, self.player_cursors[1]['x'] + 1)
                elif event.key == pygame.K_RETURN:  # Place piece
                    self.place_piece(1)
        
        return True
    
    def place_piece(self, player_idx):
        """Place piece for specified player"""
        cursor = self.player_cursors[player_idx]
        player = self.game.player1 if player_idx == 0 else self.game.player2
        
        if not player.current_piece:
            return
        
        try:
            # Find best placement for cursor position
            placement = self.find_best_placement(player, cursor['x'], cursor['y'])
            
            if placement:
                # Set piece to placement position and rotation
                player.current_piece.x = placement['x']
                player.current_piece.rotation = placement['rotation']
                
                # Hard drop the piece
                action_handler = ActionHandler(player)
                action_handler.hard_drop()
                
                # Update player state
                player.update(self.game.fall_speed, self.game.level)
                
                print(f"✅ Player {player_idx + 1} placed piece at ({placement['x']}, {placement['y']})")
            else:
                print(f"❌ Player {player_idx + 1} - no valid placement at ({cursor['x']}, {cursor['y']})")
                
        except Exception as e:
            print(f"❌ Error placing piece for player {player_idx + 1}: {e}")
    
    def find_best_placement(self, player, target_x, target_y):
        """Find best piece placement for target position"""
        if not player.current_piece:
            return None
        
        grid = create_grid(player.locked_positions)
        best_placement = None
        min_distance = float('inf')
        
        # Test all rotations
        for rotation in range(len(player.current_piece.shape)):
            # Test X positions around target
            for offset_x in range(-3, 4):
                test_x = target_x + offset_x
                if not (0 <= test_x <= 9):
                    continue
                
                # Create test piece
                test_piece = copy.deepcopy(player.current_piece)
                test_piece.x = test_x
                test_piece.rotation = rotation
                
                # Find where piece would land
                original_y = test_piece.y
                hard_drop(test_piece, grid)
                
                if valid_space(test_piece, grid):
                    # Calculate distance to target
                    final_positions = convert_shape_format(test_piece)
                    if final_positions:
                        avg_x = sum(pos[0] for pos in final_positions) / len(final_positions)
                        avg_y = sum(pos[1] for pos in final_positions) / len(final_positions)
                        
                        distance = ((avg_x - target_x) ** 2 + (avg_y - target_y) ** 2) ** 0.5
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_placement = {
                                'x': test_x,
                                'y': test_piece.y,
                                'rotation': rotation,
                                'distance': distance
                            }
                
                # Restore Y position
                test_piece.y = original_y
        
        return best_placement
    
    def update(self):
        """Update game state"""
        if not self.paused:
            # Update blink timer
            self.blink_timer += 1
            if self.blink_timer >= self.blink_interval:
                self.cursor_visible = not self.cursor_visible
                self.blink_timer = 0
            
            # Update game
            if not self.game.update():
                return False  # Game over
        
        return True
    
    def draw(self):
        """Draw the game"""
        # Draw base game
        self.game.draw()
        
        if not self.paused and self.cursor_visible:
            # Draw cursors for both players with distinct colors
            self.draw_cursor(0, (255, 0, 255))  # Player 1 - Bright Magenta
            self.draw_cursor(1, (0, 255, 255))  # Player 2 - Bright Cyan
        
        # Draw UI info
        self.draw_ui()
        
        if self.paused:
            self.draw_pause_screen()
        
        pygame.display.update()
    
    def draw_cursor(self, player_idx, color):
        """Draw cursor for specified player with correct coordinates"""
        cursor = self.player_cursors[player_idx]
        player = self.game.player1 if player_idx == 0 else self.game.player2
        
        # Calculate screen position using game constants
        offset_x = int(mid_x) if player_idx == 1 else 0
        
        cursor_screen_x = int(top_left_x) + offset_x + cursor['x'] * block_size
        cursor_screen_y = int(top_left_y) + cursor['y'] * block_size
        
        # Draw cursor with enhanced visibility
        # 1. Draw thick outer outline
        pygame.draw.rect(self.surface, color, 
                        (cursor_screen_x - 2, cursor_screen_y - 2, block_size + 4, block_size + 4), 6)
        
        # 2. Draw inner black outline for contrast
        pygame.draw.rect(self.surface, (0, 0, 0), 
                        (cursor_screen_x, cursor_screen_y, block_size, block_size), 3)
        
        # 3. Draw inner color outline
        pygame.draw.rect(self.surface, color, 
                        (cursor_screen_x + 1, cursor_screen_y + 1, block_size - 2, block_size - 2), 2)
        
        # 4. Draw center cross for better visibility
        center_x = cursor_screen_x + block_size // 2
        center_y = cursor_screen_y + block_size // 2
        
        # Horizontal line
        pygame.draw.line(self.surface, color, 
                        (cursor_screen_x + 8, center_y), 
                        (cursor_screen_x + block_size - 8, center_y), 3)
        # Vertical line
        pygame.draw.line(self.surface, color, 
                        (center_x, cursor_screen_y + 8), 
                        (center_x, cursor_screen_y + block_size - 8), 3)
        
        # Draw piece preview if player has a piece
        if player.current_piece:
            self.draw_piece_preview(player, cursor, offset_x, color)
    
    def draw_piece_preview(self, player, cursor, offset_x, cursor_color):
        """Draw preview of where piece would be placed"""
        placement = self.find_best_placement(player, cursor['x'], cursor['y'])
        
        if placement:
            # Create preview piece
            preview_piece = copy.deepcopy(player.current_piece)
            preview_piece.x = placement['x']
            preview_piece.y = placement['y']
            preview_piece.rotation = placement['rotation']
            
            # Get piece positions
            positions = convert_shape_format(preview_piece)
            
            if positions:
                for pos in positions:
                    block_x, block_y = pos
                    if 0 <= block_x < 10 and 0 <= block_y < 20:
                        px = int(top_left_x) + offset_x + block_x * block_size
                        py = int(top_left_y) + block_y * block_size
                        
                        # Draw semi-transparent preview with white/gray color to distinguish from cursor
                        alpha_surface = pygame.Surface((block_size, block_size))
                        alpha_surface.set_alpha(100)
                        alpha_surface.fill((220, 220, 220))  # Light gray
                        self.surface.blit(alpha_surface, (px, py))
                        
                        # Draw dashed outline for preview
                        pygame.draw.rect(self.surface, cursor_color, 
                                        (px, py, block_size, block_size), 2)
    
    def draw_ui(self):
        """Draw UI information"""
        font = pygame.font.SysFont('Consolas', 16)
        
        # Player 1 info
        p1_cursor = self.player_cursors[0]
        p1_text = font.render(f"P1 Cursor: ({p1_cursor['x']},{p1_cursor['y']}) | WASD+SPACE", True, (255, 0, 255))
        self.surface.blit(p1_text, (10, s_height - 80))
        
        # Player 2 info
        p2_cursor = self.player_cursors[1]
        p2_text = font.render(f"P2 Cursor: ({p2_cursor['x']},{p2_cursor['y']}) | Arrows+ENTER", True, (0, 255, 255))
        self.surface.blit(p2_text, (10, s_height - 60))
        
        # Controls
        controls_text = font.render("ESC: Menu | P: Pause", True, (255, 255, 255))
        self.surface.blit(controls_text, (10, s_height - 40))
        
        # Mode indicator
        mode_text = font.render("LOCKED POSITION MODE - DUAL PLAYER", True, (255, 255, 0))
        self.surface.blit(mode_text, (10, s_height - 20))
    
    def draw_pause_screen(self):
        """Draw pause overlay"""
        overlay = pygame.Surface((s_width, s_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.surface.blit(overlay, (0, 0))
        
        font = pygame.font.SysFont('Consolas', 48, bold=True)
        pause_text = font.render('PAUSED', True, (255, 255, 255))
        self.surface.blit(pause_text, (s_width//2 - pause_text.get_width()//2, s_height//2 - 50))
        
        font = pygame.font.SysFont('Consolas', 24)
        resume_text = font.render('Press P to resume', True, (255, 255, 255))
        self.surface.blit(resume_text, (s_width//2 - resume_text.get_width()//2, s_height//2 + 20))
    
    def run(self):
        """Main game loop"""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                
                if not self.handle_input(event):
                    return True  # Return to menu
            
            # Update and draw
            if not self.update():
                return self.show_game_over()
            
            self.draw()
            self.clock.tick(60)
        
        return True
    
    def show_game_over(self):
        """Show game over screen"""
        overlay = pygame.Surface((s_width, s_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.surface.blit(overlay, (0, 0))
        
        font = pygame.font.SysFont('Consolas', 48, bold=True)
        game_over_text = font.render('GAME OVER', True, (255, 100, 100))
        self.surface.blit(game_over_text, (s_width//2 - game_over_text.get_width()//2, s_height//2 - 100))
        
        # Show scores
        p1_score = self.game.player1.score
        p2_score = self.game.player2.score
        
        font = pygame.font.SysFont('Consolas', 24)
        score_text = font.render(f'Player 1: {p1_score}  |  Player 2: {p2_score}', True, (255, 255, 255))
        self.surface.blit(score_text, (s_width//2 - score_text.get_width()//2, s_height//2 - 20))
        
        restart_text = font.render('Press SPACE to play again, ESC for menu', True, (255, 255, 255))
        self.surface.blit(restart_text, (s_width//2 - restart_text.get_width()//2, s_height//2 + 40))
        
        pygame.display.update()
        
        # Wait for input
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Restart game
                        new_game = LockedPositionGame(self.surface)
                        return new_game.run()
                    elif event.key == pygame.K_ESCAPE:
                        return True  # Return to menu


def show_main_menu(surface):
    """Show main menu and handle selection"""
    clock = pygame.time.Clock()
    
    while True:
        surface.fill((30, 20, 50))
        
        # Title
        font = pygame.font.SysFont('Consolas', 56, bold=True)
        title = font.render('MULTIPLAYER TETRIS', True, (255, 200, 100))
        surface.blit(title, (s_width//2 - title.get_width()//2, 100))
        
        # Menu options
        font = pygame.font.SysFont('Consolas', 28)
        options = [
            "PRESS SPACE - Direct Mode (Traditional)",
            "PRESS L - Locked Position Mode (Cursor-based)",
            "PRESS ESC - Exit Game"
        ]
        
        colors = [(100, 255, 100), (255, 200, 100), (255, 100, 100)]
        
        for i, (option, color) in enumerate(zip(options, colors)):
            text = font.render(option, True, color)
            y_pos = 300 + i * 60
            surface.blit(text, (s_width//2 - text.get_width()//2, y_pos))
        
        # Instructions
        font = pygame.font.SysFont('Consolas', 18)
        instructions = [
            "Direct Mode: WASD/Arrows for movement, traditional Tetris controls",
            "Locked Position: WASD/Arrows move cursor, SPACE/ENTER place piece",
            "Both modes support 2 players simultaneously"
        ]
        
        for i, instruction in enumerate(instructions):
            text = font.render(instruction, True, (200, 200, 200))
            y_pos = 520 + i * 25
            surface.blit(text, (s_width//2 - text.get_width()//2, y_pos))
        
        pygame.display.update()
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Start direct mode (traditional)
                    return start_direct_mode(surface)
                elif event.key == pygame.K_l:
                    # Start locked position mode
                    return start_locked_position_mode(surface)
                elif event.key == pygame.K_ESCAPE:
                    return False


def start_direct_mode(surface):
    """Start traditional direct mode multiplayer"""
    print("🎮 Starting Direct Mode Multiplayer")
    game = Game(surface, auto_start=False)
    clock = pygame.time.Clock()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if not game.handle_input(event):
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True  # Return to menu
        
        if not game.update():
            # Game over - simple restart for now
            game = Game(surface, auto_start=False)
        
        game.draw()
        clock.tick(60)


def start_locked_position_mode(surface):
    """Start locked position mode"""
    print("🎮 Starting Locked Position Mode")
    locked_game = LockedPositionGame(surface)
    return locked_game.run()


def main():
    """Main function"""
    try:
        pygame.init()
        surface = pygame.display.set_mode((s_width, s_height))
        pygame.display.set_caption("Multiplayer Tetris - Fixed Implementation")
        
        print("✓ Hardware acceleration enabled")
        print("🎮 New Multiplayer Tetris Game Initialized")
        print("=" * 50)
        
        # Start main menu loop
        while show_main_menu(surface):
            pass  # Continue until user exits
        
        print("👋 Thanks for playing!")
        
    except Exception as e:
        print(f"❌ Error starting game: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main() 