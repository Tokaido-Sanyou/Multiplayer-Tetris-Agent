#!/usr/bin/env python3

"""
Multiplayer Tetris Game with Keyboard Controls
Play head-to-head Tetris with full keyboard support for 2 players
"""

import pygame
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.game.game import Game
from envs.game.constants import s_width, s_height, mid_x, mid_y

def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont('Consolas', size, bold=True, italic=True)
    label = font.render(text, True, color)
    surface.blit(label, (mid_x - label.get_width() / 2, mid_y - label.get_height() / 2))

def draw_text_bottom(text, size, color, surface):
    font = pygame.font.SysFont('Consolas', size, bold=True, italic=True)
    label = font.render(text, True, color)
    surface.blit(label, (mid_x - label.get_width() / 2, s_height - label.get_height() - 50))

def show_controls(surface):
    """Display the controls screen"""
    surface.fill((20, 20, 30))
    
    # Title
    font = pygame.font.SysFont('Consolas', 48, bold=True)
    title = font.render('TETRIS CONTROLS', True, (255, 255, 255))
    surface.blit(title, (mid_x - title.get_width() / 2, 50))
    
    # Player 1 controls
    y_offset = 150
    font = pygame.font.SysFont('Consolas', 24, bold=True)
    p1_title = font.render('PLAYER 1 (Left Side)', True, (100, 255, 100))
    surface.blit(p1_title, (50, y_offset))
    
    font = pygame.font.SysFont('Consolas', 18)
    controls_p1 = [
        "A / D        - Move left / right",
        "S            - Soft drop (move down)", 
        "W            - Rotate clockwise",
        "Q            - Rotate counter-clockwise",
        "SPACE        - Hard drop",
        "C            - Hold piece"
    ]
    
    for i, control in enumerate(controls_p1):
        text = font.render(control, True, (255, 255, 255))
        surface.blit(text, (70, y_offset + 40 + i * 25))
    
    # Player 2 controls
    font = pygame.font.SysFont('Consolas', 24, bold=True)
    p2_title = font.render('PLAYER 2 (Right Side)', True, (100, 100, 255))
    surface.blit(p2_title, (50, y_offset + 220))
    
    font = pygame.font.SysFont('Consolas', 18)
    controls_p2 = [
        "LEFT / RIGHT - Move left / right",
        "DOWN         - Soft drop (move down)",
        "UP           - Rotate clockwise", 
        "RIGHT SHIFT  - Rotate counter-clockwise",
        "ENTER        - Hard drop",
        "RIGHT CTRL   - Hold piece"
    ]
    
    for i, control in enumerate(controls_p2):
        text = font.render(control, True, (255, 255, 255))
        surface.blit(text, (70, y_offset + 260 + i * 25))
    
    # Game controls
    font = pygame.font.SysFont('Consolas', 24, bold=True)
    game_title = font.render('GAME CONTROLS', True, (255, 255, 100))
    surface.blit(game_title, (50, y_offset + 420))
    
    font = pygame.font.SysFont('Consolas', 18)
    controls_game = [
        "P            - Pause game",
        "ESC          - Return to menu",
    ]
    
    for i, control in enumerate(controls_game):
        text = font.render(control, True, (255, 255, 255))
        surface.blit(text, (70, y_offset + 460 + i * 25))
    
    # Instructions
    draw_text_bottom("PRESS ANY KEY TO RETURN TO MENU", 20, (255, 255, 255), surface)
    
    pygame.display.update()

def main_menu(surface):
    """Main menu for the multiplayer Tetris game"""
    clock = pygame.time.Clock()
    run = True
    
    while run:
        surface.fill((30, 20, 40))
        
        # Title
        font = pygame.font.SysFont('Consolas', 56, bold=True, italic=True)
        title = font.render('MULTIPLAYER TETRIS', True, (255, 200, 100))
        surface.blit(title, (mid_x - title.get_width() / 2, mid_y - 200))
        
        # Menu options
        font = pygame.font.SysFont('Consolas', 24)
        options = [
            "PRESS SPACE TO START GAME",
            "PRESS C TO VIEW CONTROLS", 
            "PRESS ESC TO EXIT"
        ]
        
        for i, option in enumerate(options):
            color = (255, 255, 255) if i != 0 else (100, 255, 100)
            text = font.render(option, True, color)
            surface.blit(text, (mid_x - text.get_width() / 2, mid_y + i * 40))
        
        # Footer
        font = pygame.font.SysFont('Consolas', 16)
        footer = font.render("Two player head-to-head Tetris with full keyboard support", True, (150, 150, 150))
        surface.blit(footer, (mid_x - footer.get_width() / 2, s_height - 100))
        
        pygame.display.update()
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return play_game(surface)
                elif event.key == pygame.K_c:
                    show_controls(surface)
                    # Wait for any key to return
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                return False
                            if event.type == pygame.KEYDOWN:
                                waiting = False
                                break
                elif event.key == pygame.K_ESCAPE:
                    run = False
                    return False
    
    return False

def play_game(surface):
    """Start and run the multiplayer game"""
    game = Game(surface, auto_start=False)
    clock = pygame.time.Clock()
    run = True
    paused = False
    
    print("🎮 Starting Multiplayer Tetris Game!")
    print("Player 1 (Green): WASD + Space + C")
    print("Player 2 (Blue): Arrow Keys + Enter + Right Ctrl")
    print("Press P to pause, ESC to return to menu")
    
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True  # Return to menu
                elif event.key == pygame.K_p:
                    paused = not paused
                    if paused:
                        # Show pause screen
                        overlay = pygame.Surface((s_width, s_height))
                        overlay.set_alpha(128)
                        overlay.fill((0, 0, 0))
                        surface.blit(overlay, (0, 0))
                        
                        font = pygame.font.SysFont('Consolas', 48, bold=True)
                        pause_text = font.render('PAUSED', True, (255, 255, 255))
                        surface.blit(pause_text, (mid_x - pause_text.get_width() / 2, mid_y - 50))
                        
                        font = pygame.font.SysFont('Consolas', 24)
                        resume_text = font.render('Press P to resume', True, (255, 255, 255))
                        surface.blit(resume_text, (mid_x - resume_text.get_width() / 2, mid_y + 20))
                        
                        pygame.display.update()
                else:
                    if not paused:
                        game.handle_input(event)
        
        if not paused:
            # Update game state
            if not game.update():
                # Game over
                return show_game_over(surface, game)
            
            # Draw game
            game.draw()
        
        clock.tick(60)
    
    return True

def show_game_over(surface, game):
    """Show game over screen"""
    # Determine winner
    p1_score = game.player1.score
    p2_score = game.player2.score
    
    overlay = pygame.Surface((s_width, s_height))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    surface.blit(overlay, (0, 0))
    
    font = pygame.font.SysFont('Consolas', 48, bold=True)
    game_over_text = font.render('GAME OVER', True, (255, 100, 100))
    surface.blit(game_over_text, (mid_x - game_over_text.get_width() / 2, mid_y - 120))
    
    # Show winner
    font = pygame.font.SysFont('Consolas', 32, bold=True)
    if p1_score > p2_score:
        winner_text = font.render('PLAYER 1 WINS!', True, (100, 255, 100))
    elif p2_score > p1_score:
        winner_text = font.render('PLAYER 2 WINS!', True, (100, 100, 255))
    else:
        winner_text = font.render('TIE GAME!', True, (255, 255, 100))
    
    surface.blit(winner_text, (mid_x - winner_text.get_width() / 2, mid_y - 60))
    
    # Show scores
    font = pygame.font.SysFont('Consolas', 24)
    score_text = font.render(f'Player 1: {p1_score}  |  Player 2: {p2_score}', True, (255, 255, 255))
    surface.blit(score_text, (mid_x - score_text.get_width() / 2, mid_y))
    
    # Instructions
    font = pygame.font.SysFont('Consolas', 20)
    restart_text = font.render('Press SPACE to play again, ESC to return to menu', True, (255, 255, 255))
    surface.blit(restart_text, (mid_x - restart_text.get_width() / 2, mid_y + 60))
    
    pygame.display.update()
    
    # Wait for input
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return play_game(surface)  # Restart game
                elif event.key == pygame.K_ESCAPE:
                    return True  # Return to menu
                    
    return True

def main():
    """Main function to start the multiplayer Tetris game"""
    try:
        # Initialize Pygame
        pygame.init()
        
        # Set up display
        surface = pygame.display.set_mode((s_width, s_height))
        pygame.display.set_caption("Multiplayer Tetris - Keyboard Controls")
        
        # Enable GPU acceleration if available
        try:
            pygame.display.set_mode((s_width, s_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            print("✓ Hardware acceleration enabled")
        except:
            print("⚠️ Hardware acceleration not available, using software rendering")
        
        print("🎮 Multiplayer Tetris Game Initialized")
        print("=" * 50)
        
        # Start main menu
        while main_menu(surface):
            pass  # Continue running until user exits
        
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