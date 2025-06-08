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
    surface.blit(title, (mid_x - title.get_width() / 2, 30))
    
    # Mode tabs
    font = pygame.font.SysFont('Consolas', 20, bold=True)
    direct_tab = font.render('DIRECT MODE', True, (100, 255, 100))
    surface.blit(direct_tab, (50, 90))
    
    locked_tab = font.render('LOCKED POSITION MODE', True, (255, 200, 100))
    surface.blit(locked_tab, (400, 90))
    
    # Direct Mode Controls
    y_offset = 120
    font = pygame.font.SysFont('Consolas', 18, bold=True)
    p1_title = font.render('PLAYER 1', True, (100, 255, 100))
    surface.blit(p1_title, (50, y_offset))
    
    font = pygame.font.SysFont('Consolas', 14)
    controls_p1_direct = [
        "A / D    - Move left/right",
        "S        - Soft drop", 
        "W        - Rotate clockwise",
        "Q        - Rotate counter-CW",
        "SPACE    - Hard drop",
        "C        - Hold piece"
    ]
    
    for i, control in enumerate(controls_p1_direct):
        text = font.render(control, True, (255, 255, 255))
        surface.blit(text, (60, y_offset + 25 + i * 20))
    
    font = pygame.font.SysFont('Consolas', 18, bold=True)
    p2_title = font.render('PLAYER 2', True, (100, 100, 255))
    surface.blit(p2_title, (50, y_offset + 170))
    
    font = pygame.font.SysFont('Consolas', 14)
    controls_p2_direct = [
        "←/→      - Move left/right",
        "↓        - Soft drop",
        "↑        - Rotate clockwise", 
        "R-SHIFT  - Rotate counter-CW",
        "ENTER    - Hard drop",
        "R-CTRL   - Hold piece"
    ]
    
    for i, control in enumerate(controls_p2_direct):
        text = font.render(control, True, (255, 255, 255))
        surface.blit(text, (60, y_offset + 195 + i * 20))
    
    # Locked Position Mode Controls
    font = pygame.font.SysFont('Consolas', 18, bold=True)
    p1_locked_title = font.render('PLAYER 1', True, (100, 255, 100))
    surface.blit(p1_locked_title, (400, y_offset))
    
    font = pygame.font.SysFont('Consolas', 14)
    controls_p1_locked = [
        "W/A/S/D  - Navigate grid",
        "SPACE    - Place at position",
        "Q        - Rotate piece",
        "C        - Hold piece"
    ]
    
    for i, control in enumerate(controls_p1_locked):
        text = font.render(control, True, (255, 255, 255))
        surface.blit(text, (410, y_offset + 25 + i * 20))
    
    font = pygame.font.SysFont('Consolas', 18, bold=True)
    p2_locked_title = font.render('PLAYER 2', True, (100, 100, 255))
    surface.blit(p2_locked_title, (400, y_offset + 120))
    
    font = pygame.font.SysFont('Consolas', 14)
    controls_p2_locked = [
        "↑/←/↓/→  - Navigate grid",
        "ENTER    - Place at position",
        "R-SHIFT  - Rotate piece",
        "R-CTRL   - Hold piece"
    ]
    
    for i, control in enumerate(controls_p2_locked):
        text = font.render(control, True, (255, 255, 255))
        surface.blit(text, (410, y_offset + 145 + i * 20))
    
    # Game controls
    font = pygame.font.SysFont('Consolas', 20, bold=True)
    game_title = font.render('GAME CONTROLS', True, (255, 255, 100))
    surface.blit(game_title, (50, y_offset + 320))
    
    font = pygame.font.SysFont('Consolas', 16)
    controls_game = [
        "P            - Pause game",
        "ESC          - Return to menu",
        "TAB          - Toggle mode display"
    ]
    
    for i, control in enumerate(controls_game):
        text = font.render(control, True, (255, 255, 255))
        surface.blit(text, (70, y_offset + 350 + i * 25))
    
    # Instructions
    draw_text_bottom("PRESS ANY KEY TO RETURN TO MENU", 18, (255, 255, 255), surface)
    
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
            "PRESS SPACE TO START DIRECT MODE",
            "PRESS L TO START LOCKED POSITION MODE",
            "PRESS C TO VIEW CONTROLS", 
            "PRESS ESC TO EXIT"
        ]
        
        for i, option in enumerate(options):
            if i == 0:
                color = (100, 255, 100)
            elif i == 1:
                color = (255, 200, 100)
            else:
                color = (255, 255, 255)
            text = font.render(option, True, color)
            surface.blit(text, (mid_x - text.get_width() / 2, mid_y + i * 40))
        
        # Mode descriptions
        font = pygame.font.SysFont('Consolas', 16)
        desc1 = font.render("Direct: Traditional controls (WASD/Arrows for movement)", True, (150, 150, 150))
        surface.blit(desc1, (mid_x - desc1.get_width() / 2, mid_y + 180))
        
        desc2 = font.render("Locked Position: Navigate grid positions (WASD/Arrows to select placement)", True, (150, 150, 150))
        surface.blit(desc2, (mid_x - desc2.get_width() / 2, mid_y + 200))
        
        pygame.display.update()
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return play_game(surface, mode='direct')
                elif event.key == pygame.K_l:
                    return play_game(surface, mode='locked_position')
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

def play_game(surface, mode='direct'):
    """Start and run the multiplayer game"""
    game = Game(surface, auto_start=False)
    clock = pygame.time.Clock()
    run = True
    paused = False
    
    # Position selection state for locked position mode
    position_cursors = [{'x': 4, 'y': 19}, {'x': 4, 'y': 19}]  # Start at bottom-center
    
    print(f"🎮 Starting Multiplayer Tetris Game ({mode.upper()} MODE)!")
    if mode == 'direct':
        print("Player 1 (Green): WASD + Space + C")
        print("Player 2 (Blue): Arrow Keys + Enter + Right Ctrl")
    else:
        print("Player 1 (Green): WASD to navigate, Space to place, Q to rotate, C to hold")
        print("Player 2 (Blue): Arrow Keys to navigate, Enter to place, Right Shift to rotate, Right Ctrl to hold")
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
                        if mode == 'direct':
                            game.handle_input(event)
                        elif mode == 'locked_position':
                            handle_locked_position_input(event, game, position_cursors)
        
        if not paused:
            # Update game state
            if not game.update():
                # Game over
                return show_game_over(surface, game)
            
            # Draw game
            game.draw()
            
            # Draw position cursors for locked position mode
            if mode == 'locked_position':
                draw_position_cursors(surface, game, position_cursors)
        
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
                    return play_game(surface, mode='direct')  # Restart game in direct mode
                elif event.key == pygame.K_ESCAPE:
                    return True  # Return to menu
                    
    return True

def handle_locked_position_input(event, game, position_cursors):
    """Handle input for locked position mode"""
    from envs.tetris_env import TetrisEnv
    
    # Player 1 controls (WASD + Space + Q + C)
    if event.key == pygame.K_w:  # Move cursor up
        position_cursors[0]['y'] = max(0, position_cursors[0]['y'] - 1)
    elif event.key == pygame.K_s:  # Move cursor down
        position_cursors[0]['y'] = min(19, position_cursors[0]['y'] + 1)
    elif event.key == pygame.K_a:  # Move cursor left
        position_cursors[0]['x'] = max(0, position_cursors[0]['x'] - 1)
    elif event.key == pygame.K_d:  # Move cursor right
        position_cursors[0]['x'] = min(9, position_cursors[0]['x'] + 1)
    elif event.key == pygame.K_SPACE:  # Place piece at cursor position
        execute_locked_placement(game.player1, position_cursors[0], game)
    elif event.key == pygame.K_q:  # Rotate piece
        if game.player1.current_piece:
            game.player1.current_piece.rotation = (game.player1.current_piece.rotation + 1) % 4
    elif event.key == pygame.K_c:  # Hold piece
        try:
            from envs.game.action_handler import ActionHandler
            action_handler = ActionHandler(game.player1)
            action_handler.hold_piece()
        except:
            pass
    
    # Player 2 controls (Arrow keys + Enter + Right Shift + Right Ctrl)
    elif event.key == pygame.K_UP:  # Move cursor up
        position_cursors[1]['y'] = max(0, position_cursors[1]['y'] - 1)
    elif event.key == pygame.K_DOWN:  # Move cursor down
        position_cursors[1]['y'] = min(19, position_cursors[1]['y'] + 1)
    elif event.key == pygame.K_LEFT:  # Move cursor left
        position_cursors[1]['x'] = max(0, position_cursors[1]['x'] - 1)
    elif event.key == pygame.K_RIGHT:  # Move cursor right
        position_cursors[1]['x'] = min(9, position_cursors[1]['x'] + 1)
    elif event.key == pygame.K_RETURN:  # Place piece at cursor position
        execute_locked_placement(game.player2, position_cursors[1], game)
    elif event.key == pygame.K_RSHIFT:  # Rotate piece
        if game.player2.current_piece:
            game.player2.current_piece.rotation = (game.player2.current_piece.rotation + 1) % 4
    elif event.key == pygame.K_RCTRL:  # Hold piece
        try:
            from envs.game.action_handler import ActionHandler
            action_handler = ActionHandler(game.player2)
            action_handler.hold_piece()
        except:
            pass

def execute_locked_placement(player, cursor_pos, game_context):
    """Execute piece placement at cursor position"""
    if not player.current_piece:
        return
    
    try:
        # Use the provided game context instead of creating temporary environment
        from envs.game.action_handler import ActionHandler
        
        action_handler = ActionHandler(player)
        current_piece = player.current_piece
        
        # Find best placement for the target position
        placement = find_best_placement_for_position(player, cursor_pos['x'], cursor_pos['y'])
        
        if placement:
            # Set the piece to the target configuration
            current_piece.x = placement['x']
            current_piece.rotation = placement['rotation']
            
            # Hard drop to place the piece
            action_handler.hard_drop()
            
            # Update player with game context
            player.update(game_context.fall_speed, game_context.level)
        
    except Exception as e:
        print(f"Error in locked placement: {e}")
        # Fallback to simple hard drop
        try:
            from envs.game.action_handler import ActionHandler
            action_handler = ActionHandler(player)
            action_handler.hard_drop()
        except:
            pass

def find_best_placement_for_position(player, target_x, target_y):
    """Find the best placement for a piece near the target position"""
    if not player.current_piece:
        return None
    
    try:
        from envs.game.utils import create_grid, hard_drop
        from envs.game.piece_utils import valid_space, convert_shape_format
        import copy
    except ImportError:
        from game.utils import create_grid, hard_drop
        from game.piece_utils import valid_space, convert_shape_format
        import copy
    
    grid = create_grid(player.locked_positions)
    best_placement = None
    min_distance = float('inf')
    
    # Test different rotations and x positions
    for rotation in range(4):
        for test_x in range(-2, 12):  # Allow some positions outside grid for edge pieces
            # Create a test piece
            test_piece = copy.deepcopy(player.current_piece)
            test_piece.x = test_x
            test_piece.rotation = rotation
            
            # Hard drop to find final position
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
            
            # Restore original position
            test_piece.y = original_y
    
    return best_placement

def draw_position_cursors(surface, game, position_cursors):
    """Draw position selection cursors for locked position mode with blinking effect"""
    import time
    
    # Grid cell size (approximately)
    cell_size = 30
    grid_start_x = 25  # Approximate grid start position
    grid_start_y = 30
    
    # Blinking effect - alternate visibility every 500ms
    blink_time = int(time.time() * 1000) % 1000  # 1000ms cycle
    show_cursor = blink_time < 500  # Show for first 500ms, hide for next 500ms
    
    if show_cursor:
        # Player 1 cursor (green)
        cursor1 = position_cursors[0]
        cursor1_x = grid_start_x + cursor1['x'] * cell_size
        cursor1_y = grid_start_y + cursor1['y'] * cell_size
        
        # Draw green cursor with preview of piece placement
        draw_placement_preview(surface, game.player1, cursor1, cursor1_x, cursor1_y, cell_size, (0, 255, 0))
        
        # Player 2 cursor (blue) - offset for second player
        cursor2 = position_cursors[1]
        player2_offset = s_width // 2  # Second player grid offset
        cursor2_x = grid_start_x + player2_offset + cursor2['x'] * cell_size
        cursor2_y = grid_start_y + cursor2['y'] * cell_size
        
        # Draw blue cursor with preview of piece placement
        draw_placement_preview(surface, game.player2, cursor2, cursor2_x, cursor2_y, cell_size, (0, 0, 255))
    
    # Draw position indicators (always visible)
    font = pygame.font.SysFont('Consolas', 12)
    
    # Player 1 position text
    cursor1 = position_cursors[0]
    pos1_text = font.render(f"P1: ({cursor1['x']},{cursor1['y']})", True, (0, 255, 0))
    surface.blit(pos1_text, (10, s_height - 80))
    
    # Player 2 position text
    cursor2 = position_cursors[1]
    pos2_text = font.render(f"P2: ({cursor2['x']},{cursor2['y']})", True, (0, 0, 255))
    surface.blit(pos2_text, (10, s_height - 60))
    
    # Mode indicator
    mode_text = font.render("LOCKED POSITION MODE", True, (255, 255, 0))
    surface.blit(mode_text, (10, s_height - 40))

def draw_placement_preview(surface, player, cursor_pos, screen_x, screen_y, cell_size, cursor_color):
    """Draw a preview of where the piece would be placed"""
    if not player.current_piece:
        # Just draw cursor outline if no piece
        pygame.draw.rect(surface, cursor_color, 
                        (screen_x, screen_y, cell_size, cell_size), 3)
        return
    
    # Find best placement for this cursor position
    placement = find_best_placement_for_position(player, cursor_pos['x'], cursor_pos['y'])
    
    if placement:
        # Draw the piece preview at the final placement position
        try:
            from envs.game.piece_utils import convert_shape_format
            import copy
        except ImportError:
            from game.piece_utils import convert_shape_format
            import copy
        
        # Create a preview piece at the final position
        preview_piece = copy.deepcopy(player.current_piece)
        preview_piece.x = placement['x']
        preview_piece.y = placement['y']
        preview_piece.rotation = placement['rotation']
        
        # Get the piece shape positions
        shape_positions = convert_shape_format(preview_piece)
        
        if shape_positions:
            # Draw semi-transparent preview blocks
            preview_color = (*cursor_color, 128)  # Add alpha for transparency
            
            for pos in shape_positions:
                block_x, block_y = pos
                if 0 <= block_x < 10 and 0 <= block_y < 20:
                    # Calculate screen position (adjust for grid layout)
                    grid_start_x = 25
                    grid_start_y = 30
                    
                    # Adjust for player 2 offset if needed
                    offset_x = 0
                    if cursor_color == (0, 0, 255):  # Player 2 (blue)
                        offset_x = s_width // 2
                    
                    px = grid_start_x + offset_x + block_x * cell_size
                    py = grid_start_y + block_y * cell_size
                    
                    # Create a surface with alpha for transparency
                    alpha_surface = pygame.Surface((cell_size, cell_size))
                    alpha_surface.set_alpha(128)
                    alpha_surface.fill(cursor_color)
                    surface.blit(alpha_surface, (px, py))
                    
                    # Draw outline
                    pygame.draw.rect(surface, cursor_color, 
                                    (px, py, cell_size, cell_size), 2)
        
        # Draw current piece at cursor position for selection visualization
        draw_piece_at_cursor(surface, player, cursor_pos, screen_x, screen_y, cell_size, cursor_color)
    else:
        # No valid placement - just show cursor
        pygame.draw.rect(surface, cursor_color, 
                        (screen_x, screen_y, cell_size, cell_size), 3)

def draw_piece_at_cursor(surface, player, cursor_pos, screen_x, screen_y, cell_size, cursor_color):
    """Draw the current piece shape at the cursor position for selection visualization"""
    if not player.current_piece:
        return
    
    try:
        from envs.game.piece_utils import convert_shape_format
        import copy
    except ImportError:
        from game.piece_utils import convert_shape_format
        import copy
    
    # Create a piece at cursor position (not dropped)
    cursor_piece = copy.deepcopy(player.current_piece)
    cursor_piece.x = cursor_pos['x']
    cursor_piece.y = cursor_pos['y']
    
    # Get the shape positions
    shape_positions = convert_shape_format(cursor_piece)
    
    if shape_positions:
        # Grid layout parameters
        grid_start_x = 25
        grid_start_y = 30
        
        # Adjust for player 2 offset if needed
        offset_x = 0
        if cursor_color == (0, 0, 255):  # Player 2 (blue)
            offset_x = s_width // 2
        
        # Draw piece blocks at cursor position
        for pos in shape_positions:
            block_x, block_y = pos
            if 0 <= block_x < 10 and 0 <= block_y < 20:
                px = grid_start_x + offset_x + block_x * cell_size
                py = grid_start_y + block_y * cell_size
                
                # Draw semi-transparent piece block
                alpha_surface = pygame.Surface((cell_size, cell_size))
                alpha_surface.set_alpha(100)  # More transparent than final placement
                alpha_surface.fill(cursor_color)
                surface.blit(alpha_surface, (px, py))
                
                # Draw dashed outline for selection indication
                draw_dashed_outline(surface, pygame.Rect(px, py, cell_size, cell_size), cursor_color, 2)

def draw_dashed_outline(surface, rect, color, width):
    """Draw a dashed outline rectangle"""
    # Draw dashes on each side
    dash_length = 5
    gap_length = 3
    
    # Top edge
    x = rect.x
    while x < rect.x + rect.width:
        end_x = min(x + dash_length, rect.x + rect.width)
        pygame.draw.line(surface, color, (x, rect.y), (end_x, rect.y), width)
        x += dash_length + gap_length
    
    # Bottom edge
    x = rect.x
    while x < rect.x + rect.width:
        end_x = min(x + dash_length, rect.x + rect.width)
        pygame.draw.line(surface, color, (x, rect.y + rect.height - width), (end_x, rect.y + rect.height - width), width)
        x += dash_length + gap_length
    
    # Left edge
    y = rect.y
    while y < rect.y + rect.height:
        end_y = min(y + dash_length, rect.y + rect.height)
        pygame.draw.line(surface, color, (rect.x, y), (rect.x, end_y), width)
        y += dash_length + gap_length
    
    # Right edge
    y = rect.y
    while y < rect.y + rect.height:
        end_y = min(y + dash_length, rect.y + rect.height)
        pygame.draw.line(surface, color, (rect.x + rect.width - width, y), (rect.x + rect.width - width, end_y), width)
        y += dash_length + gap_length

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