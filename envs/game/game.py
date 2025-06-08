import pygame
import random
import collections
import os
import sys

# Handle both direct execution and module import
try:
    from .constants import *
    from .block_pool import BlockPool
    from .player import Player
    from .utils import create_grid, check_lost, draw_window, draw_next_pieces, draw_hold_piece, add_garbage_line, hard_drop, get_shape_from_index
    from .piece_utils import valid_space, convert_shape_format
    from .piece import Piece
    from .action_handler import ActionHandler
except ImportError:
    # Direct execution - imports without relative paths
    from constants import *
    from block_pool import BlockPool
    from player import Player
    from utils import create_grid, check_lost, draw_window, draw_next_pieces, draw_hold_piece, add_garbage_line, hard_drop, get_shape_from_index
    from piece_utils import valid_space, convert_shape_format
    from piece import Piece
    from action_handler import ActionHandler

"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

pygame.init()
clock = pygame.time.Clock()

# global vars
# screen(window) width and height
s_width = 1400
s_height = 700

# play section 
play_width = 300  # meaning 300 // 10 = 30 width per block
play_height = 600  # meaning 600 // 20 = 20 height per block
block_size = 30

# x and y coordinates for top-left position of play section
top_left_x = (s_width/2 - play_width) // 2
top_left_y = s_height - play_height

# coordinates for middle of screen
# use as parameter for second grid
mid_x = (s_width/2)
mid_y = (s_height/2)

# SHAPE FORMATS (SRS spawn states)
S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['.....',
      '.....',
      '0000.',
      '.....',
      '.....'],
     ['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

# Wall kick data for SRS
# Format: [current_rotation][target_rotation] = [(x, y) offsets]
# For J, L, S, T, Z pieces
JLSTZ_WALL_KICKS = {
    # 0 -> R
    0: {
        1: [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        3: [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)]  # 0 -> L
    },
    # R -> 2 or 0
    1: {
        2: [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        0: [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)]  # R -> 0
    },
    # 2 -> L or R
    2: {
        3: [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        1: [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)]  # 2 -> R
    },
    # L -> 0 or 2
    3: {
        0: [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        2: [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)]  # L -> 2
    }
}

# For I piece
I_WALL_KICKS = {
    # 0 -> R or L
    0: {
        1: [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
        3: [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)]  # 0 -> L
    },
    # R -> 2 or 0
    1: {
        2: [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
        0: [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)]  # R -> 0
    },
    # 2 -> L or R
    2: {
        3: [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
        1: [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)]  # 2 -> R
    },
    # L -> 0 or 2
    3: {
        0: [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
        2: [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)]  # L -> 2
    }
}

# list to store all shapes
shapes = [S, Z, I, O, J, L, T]
# list to store shapes' colors
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (230,230,250)]

# create a class for Piece object


def info_page(surface):
    info = True

    while info:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    info = False

        surface.fill((0,0,0))
        pygame.draw.line(surface, (255, 255, 255), (s_width / 2, 75), (s_width / 2, s_height-40))
        pygame.draw.line(surface, (255, 255, 255), (0, 75), (s_width, 75))
        pygame.draw.line(surface, (255, 255, 255), (0, s_height), (s_width, s_height-40))
        
        # Use smaller fonts
        title_font = pygame.font.SysFont('Consolas', 40, bold=True,italic=True)
        font = pygame.font.SysFont('Consolas', 18, bold=True,italic=True)
        small_font = pygame.font.SysFont('Consolas', 16, bold=True, italic=True)

        info_text = title_font.render("INFO",True,(255,255,255))
        surface.blit(info_text,(mid_x-info_text.get_width()/2,20))

        quit_text = small_font.render("( PRESS Q TO QUIT INFO PAGE )",True,(255,255,255))
        surface.blit(quit_text,(mid_x-quit_text.get_width()/2,s_height-30))

        p1_head_text = title_font.render("PLAYER 1 (LEFT)",True,(255,255,255))
        p1_info_text_1 = font.render("W  - ROTATE CW",True,(255,255,255))
        p1_info_text_2 = font.render("Q  - ROTATE CCW", True, (255, 255, 255))
        p1_info_text_3 = font.render("A - MOVE LEFT", True, (255, 255, 255))
        p1_info_text_4 = font.render("S - MOVE DOWN", True, (255, 255, 255))
        p1_info_text_5 = font.render("D - MOVE RIGHT", True, (255, 255, 255))
        p1_info_text_6 = font.render("SPACE - HARD DROP", True, (255, 255, 255))
        p1_info_text_7 = font.render("C - HOLD", True, (255, 255, 255))

        surface.blit(p1_head_text,(mid_x/2-p1_head_text.get_width()/2,mid_y/2))
        surface.blit(p1_info_text_1, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width()/32, mid_y / 2 + p1_info_text_1.get_height()*2))
        surface.blit(p1_info_text_2, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32,mid_y / 2 + p1_info_text_1.get_height() * 3.5))
        surface.blit(p1_info_text_3, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32,mid_y / 2 + p1_info_text_1.get_height() * 5))
        surface.blit(p1_info_text_4, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32,mid_y / 2 + p1_info_text_1.get_height() * 6.5))
        surface.blit(p1_info_text_5, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32,mid_y / 2 + p1_info_text_1.get_height() * 8))
        surface.blit(p1_info_text_6, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32,mid_y / 2 + p1_info_text_1.get_height() * 9.5))
        surface.blit(p1_info_text_7, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32,mid_y / 2 + p1_info_text_1.get_height() * 11))

        p2_head_text = title_font.render("PLAYER 2 (RIGHT)", True, (255, 255, 255))
        p2_info_text_1 = font.render("UP  - ROTATE CW", True, (255, 255, 255))
        p2_info_text_2 = font.render("RSHIFT  - ROTATE CCW", True, (255, 255, 255))
        p2_info_text_3 = font.render("LEFT - MOVE LEFT", True, (255, 255, 255))
        p2_info_text_4 = font.render("DOWN - MOVE DOWN", True, (255, 255, 255))
        p2_info_text_5 = font.render("RIGHT - MOVE RIGHT", True, (255, 255, 255))
        p2_info_text_6 = font.render("ENTER - HARD DROP", True, (255, 255, 255))
        p2_info_text_7 = font.render("RCTRL - HOLD", True, (255, 255, 255))

        surface.blit(p2_head_text, (mid_x / 2 - p1_head_text.get_width() / 2 + mid_x - 50, mid_y / 2))
        surface.blit(p2_info_text_1, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32 + mid_x -50,mid_y / 2 + p1_info_text_1.get_height() * 2))
        surface.blit(p2_info_text_2, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32 + mid_x -50,mid_y / 2 + p1_info_text_1.get_height() * 3.5))
        surface.blit(p2_info_text_3, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32 + mid_x -50,mid_y / 2 + p1_info_text_1.get_height() * 5))
        surface.blit(p2_info_text_4, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32 + mid_x -50,mid_y / 2 + p1_info_text_1.get_height() * 6.5))
        surface.blit(p2_info_text_5, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32 + mid_x -50,mid_y / 2 + p1_info_text_1.get_height() * 8))
        surface.blit(p2_info_text_6, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32 + mid_x -50,mid_y / 2 + p1_info_text_1.get_height() * 9.5))
        surface.blit(p2_info_text_7, (mid_x / 2 - p1_info_text_1.get_width() / 2 - p1_info_text_1.get_width() / 32 + mid_x -50,mid_y / 2 + p1_info_text_1.get_height() * 11))

        pygame.display.update()

    surface.fill((0, 0, 0))
    resumeFont = pygame.font.SysFont('Consolas', 30, bold=True, italic=True)
    resumeText = resumeFont.render("BACK TO MAIN MENU......", True, (255, 255, 255))
    surface.blit(resumeText, ((mid_x - resumeText.get_width() / 2, mid_y - resumeText.get_height() / 2)))
    pygame.display.update()
    pygame.time.delay(1000)

def pause(surface, clock):
    paused = True

    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = False
                elif event.key == pygame.K_q:
                    pygame.quit()
                    quit()

        surface.fill((0,0,0))
        pauseFont = pygame.font.SysFont('Consolas', 50, bold=True,italic=True)
        pauseText = pauseFont.render("PAUSE", True, (255, 255, 255))
        instructionFont = pygame.font.SysFont('Consolas', 20, bold=True, italic=True)
        instructionText = instructionFont.render("Press SPACE to continue. Press Q to quit.",True,(255,255,255))
        surface.blit(pauseText, (mid_x - pauseText.get_width()/2 , mid_y - pauseText.get_height()/2))
        surface.blit(instructionText, ((mid_x - instructionText.get_width()/2, mid_y + instructionText.get_height()/2 + 150 )))

        pygame.display.update()
        clock.tick(5)

    surface.fill((0,0,0))
    resumeFont = pygame.font.SysFont('Consolas', 30, bold=True, italic=True)
    resumeText = resumeFont.render("RESUME IN 2 SECONDS...", True, (255, 255, 255))
    surface.blit(resumeText, ((mid_x - resumeText.get_width()/2 , mid_y - resumeText.get_height()/2)))
    pygame.display.update()
    pygame.time.delay(2000)
    clock.tick(5)

def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont("Consolas",size,bold=True)
    label = font.render(str(text),True,color)
    surface.blit(label,(top_left_x + play_width/2 - label.get_width()/2, top_left_y + play_height/2 - label.get_height()/2))

def draw_text_bottom(text, size, color, surface):
    font = pygame.font.SysFont("Consolas",size,bold=True,italic=True)
    label = font.render(str(text),True,color)
    surface.blit(label,(mid_x - label.get_width()/2, mid_y + label.get_height()/2 + 250))

def draw_grid(surface,grid_1,grid_2,add=0):
    for i in range(len(grid_1)):
        pygame.draw.line(surface,(128,128,128),(top_left_x ,top_left_y + i*block_size),(top_left_x+play_width , top_left_y+ i * block_size))
        for j in range(len(grid_1[i])):
            pygame.draw.line(surface, (128, 128, 128), (top_left_x + j*block_size, top_left_y),(top_left_x  + j*block_size, top_left_y + play_height))

    for i in range(len(grid_2)):
        pygame.draw.line(surface,(128,128,128),(top_left_x + add,top_left_y + i*block_size),(top_left_x+play_width + add, top_left_y+ i * block_size))
        for j in range(len(grid_2[i])):
            pygame.draw.line(surface, (128, 128, 128), (top_left_x + add + j*block_size, top_left_y),(top_left_x + add + j*block_size, top_left_y + play_height))

def clear_rows(grid, locked_pos):
    inc = 0
    rows_to_clear = []
    
    # Find all complete rows
    for i in range(len(grid)-1,-1,-1):
        row = grid[i]
        if (0,0,0) not in row:
            rows_to_clear.append(i)
            inc += 1
    
    # Remove blocks from complete rows
    for row_index in rows_to_clear:
        for j in range(len(grid[row_index])):
            try:
                del locked_pos[(j, row_index)]
            except:
                continue
    
    # Move remaining blocks down
    if inc > 0:
        # Find the highest cleared row (lowest index)
        highest_cleared_row = min(rows_to_clear) if rows_to_clear else len(grid)
        
        for key in sorted(list(locked_pos), key=lambda x: x[1], reverse=True):
            x, y = key
            if y < highest_cleared_row:
                newKey = (x, y + inc)
                locked_pos[newKey] = locked_pos.pop(key)

    return inc

def add_garbage_line(locked_positions, num_lines=1):
    new_positions = {}
    
    for y in range(19, -1, -1):
        for x in range(10):
            if (x, y) in locked_positions:
                new_y = y - num_lines
                if new_y >= 0:
                    new_positions[(x, new_y)] = locked_positions[(x, y)]
    
    for i in range(num_lines):
        gap_pos = random.randint(0, 9)
        for x in range(10):
            if x != gap_pos:
                new_positions[(x, 19 - i)] = (128, 128, 128)
    
    locked_positions.clear()
    locked_positions.update(new_positions)

class Game:
    def __init__(self, surface, auto_start=False):
        self.surface = surface
        self.clock = pygame.time.Clock()
        self.fall_time = 0
        self.level_time = 0
        self.level = 1
        self.fall_speed = 0.5  # natural fall interval now 500 ms (0.5 s)
        self.auto_start = auto_start  # New flag to control auto-start behavior
        
        self.block_pool = BlockPool()
        
        self.player1 = Player(self.block_pool, True)
        self.player2 = Player(self.block_pool, False)
        
        self.p1_grid = create_grid(self.player1.locked_positions)
        self.p2_grid = create_grid(self.player2.locked_positions)
        # For incremental rendering of falling pieces
        self.prev_shape_pos_1 = []
        self.prev_shape_pos_2 = []
        self.first_draw = True
    
    def handle_input(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                pause(self.surface, self.clock)
            else:
                self.player1.handle_input(event)
                self.player2.handle_input(event)
        return True
    
    def update(self):
        self.fall_time += self.clock.get_rawtime()
        self.level_time += self.clock.get_rawtime()
        self.clock.tick()
        
        if self.level_time / 1000 > 15:
            self.level_time = 0
            if self.fall_speed > 0.12:
                self.fall_speed -= 0.01
                self.level += 1
        
        if self.fall_time/1000 >= self.fall_speed:
            self.fall_time = 0
            # Move player1 piece down
            self.player1.current_piece.y += 1
            # Move player2 piece down if present
            if self.player2.current_piece is not None:
                self.player2.current_piece.y += 1
        
        self.update_player(self.player1, self.player2)
        self.update_player(self.player2, self.player1)
        
        self.p1_grid = create_grid(self.player1.locked_positions)
        self.p2_grid = create_grid(self.player2.locked_positions)
        
        # Collision for player1
        if not valid_space(self.player1.current_piece, self.p1_grid) and self.player1.current_piece.y > 0:
            self.player1.current_piece.y -= 1
            self.player1.change_piece = True
        
        # Collision for player2, if present
        if self.player2.current_piece is not None:
            if not valid_space(self.player2.current_piece, self.p2_grid) and self.player2.current_piece.y > 0:
                self.player2.current_piece.y -= 1
                self.player2.change_piece = True
        
        if check_lost(self.player1.locked_positions) or check_lost(self.player2.locked_positions):
            return False
        
        return True
    
    def update_player(self, player, opponent):
        lines_cleared = player.update(self.fall_speed, self.level)
        if lines_cleared > 0:
            garbage_lines = self.get_garbage_lines(lines_cleared)
            if garbage_lines > 0:
                add_garbage_line(opponent.locked_positions, garbage_lines)
    
    def get_garbage_lines(self, lines_cleared):
        if lines_cleared == 1:
            return 0
        elif lines_cleared == 2:
            return 1
        elif lines_cleared == 3:
            return 2
        elif lines_cleared == 4:
            return 4
        return 0
    
    def draw(self):
        # Full redraw of grid and UI each frame for proper piece movement
        self.surface.fill((33,29,29))
        draw_window(self.surface, self.p1_grid, self.p2_grid,
                    self.player1.current_piece, self.player2.current_piece,
                    self.player1.score, self.player2.score,
                    self.level, self.fall_speed, add=int(mid_x))
        draw_next_pieces(self.player1.next_pieces, self.surface, 0)
        draw_next_pieces(self.player2.next_pieces, self.surface, 1)
        draw_hold_piece(self.player1.hold_piece, self.surface, 0)
        draw_hold_piece(self.player2.hold_piece, self.surface, 1)
        pygame.display.update()

    def start(self):
        """Start the game loop only if auto_start is True"""
        if self.auto_start:
            run = True
            while run:
                for event in pygame.event.get():
                    run = self.handle_input(event)
                run = self.update()
                self.draw()

def main(win):
    """
    Instantiate Game with auto_start=True and launch the main loop.
    """
    game = Game(win, auto_start=True)
    game.start()

def main_menu():
    run = True
    # Load background image once
    if os.path.exists('bgp.jpg'):
        try:
            background = pygame.image.load('bgp.jpg')
        except Exception as e:
            print(f'Error loading background image: {e}')
            background = None
    else:
        print('Warning: bgp.jpg not found, background will not be loaded.')
        background = None
    while run:
        win.fill((255, 178, 102))
        if background:
            win.blit(background, (0, -70))
        font = pygame.font.SysFont('Consolas', 40, bold=True, italic=True)
        label = font.render('T E T R I S', True, (255, 255, 255))
        win.blit(label, (mid_x - label.get_width() / 2, mid_y - label.get_height() / 2 - 250))
        draw_text_bottom("PRESS ANY KEY TO START THE GAME", 40, (255, 255, 255), win)

        small_font = pygame.font.SysFont('Consolas', 20, bold=True, italic=True)
        small_label = small_font.render('( PRESS I FOR INFO )', True, (255, 255, 255))
        win.blit(small_label, (mid_x - small_label.get_width() / 2, mid_y + small_label.get_height() / 2 + 310))

        pygame.display.update()
        clock.tick(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    info_page(win)
                else:
                    win.fill((0, 0, 0))
                    wait_font = pygame.font.SysFont('Consolas', 40, bold=True, italic=True)
                    wait_text = wait_font.render('STARTING GAME......', True, (255, 255, 255))
                    win.blit(wait_text, (mid_x - wait_text.get_width() / 2, mid_y))
                    pygame.display.update()
                    pygame.time.delay(1500)
                    main(win)

    pygame.display.quit()

if __name__ == '__main__':
    win = pygame.display.set_mode((s_width,s_height))
    pygame.display.set_caption("Tetris Game")
    main_menu()