import pygame
import random
import collections
from constants import *
from block_pool import BlockPool
from player import Player
from utils import create_grid, check_lost, draw_window, draw_next_pieces, draw_hold_piece, add_garbage_line

"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

pygame.init()

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
class Piece:
    def __init__(self,x,y,shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.rotation = 0
        self.color = shape_colors[shapes.index(shape)]
    
    def rotate(self, direction, grid):
        """Rotate the piece with SRS wall kicks.
        direction: 1 for clockwise, -1 for counter-clockwise
        Returns True if rotation was successful, False otherwise"""
        old_rotation = self.rotation
        new_rotation = (self.rotation + direction) % len(self.shape)
        
        # Get the appropriate wall kick data
        if self.shape == I:
            wall_kicks = I_WALL_KICKS
        elif self.shape == O:
            # O piece doesn't need wall kicks
            self.rotation = new_rotation
            return True
        else:
            wall_kicks = JLSTZ_WALL_KICKS
        
        # Try each wall kick offset
        for x_offset, y_offset in wall_kicks[old_rotation][new_rotation]:
            self.x += x_offset
            self.y += y_offset
            self.rotation = new_rotation
            
            if valid_space(self, grid):
                return True
            
            # If the position is invalid, revert the changes
            self.x -= x_offset
            self.y -= y_offset
            self.rotation = old_rotation
        
        return False

class ActionHandler:
    """Handles all possible actions in the game"""
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
                self.player.current_piece.x, self.player.current_piece.y = 3, 0  # Reset position for held piece
            self.player.can_hold = False

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
        pygame.draw.line(surface, (255, 255, 255), (0, s_height-40), (s_width, s_height-40))
        
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

def create_grid(locked_positions={}):
    # create a blank (black) 2-d array grid
    #            x-part                       y-part
    grid = [[(0,0,0) for _ in range(10)] for _ in range(20)]

    # locked_positions --> a dict which contains colors(values) of position of tetris blocks(keys) that already in the grid
    #               pos     color
    # example --> {(1,1):(0,255,255)}

    # section of code to detect and update color of grid based on locked positions
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            # (j,i) --> j is x-coordinate , i is y-coordinate
            if (j,i) in locked_positions:
                c = locked_positions[(j,i)]
                grid[i][j] = c

    return grid

def convert_shape_format(shape):
    # shape --> Piece object
    positions = []

    # to determine the which rotation of shape format to be used
    format = shape.shape[shape.rotation % (len(shape.shape))]

    # loop through list of lists to locate the position of '0'
    for i,line in enumerate(format):
        for j, column in enumerate(line):
            if column == '0':
                # let the positions stick with the shape x and y coordinates
                positions.append((shape.x+j,shape.y+i))

    # to offset(cancel off)/adjust the position displayed
    for i,pos in enumerate(positions):
        positions[i] = (pos[0], pos[1])  # Removed offset adjustment

    return positions

def valid_space(shape, grid):
    # create a list of accepted positions from (0,0) to (9,19) and only if the position is not occupied by existed block (color equal to (0,0,0))
    accepted_pos = [(j,i) for j in range(10) for i in range(20) if grid[i][j] == (0,0,0)]

    formatted = convert_shape_format(shape)

    for pos in formatted:
        x, y = pos
        # Check if position is within grid boundaries
        if x < 0 or x >= 10 or y >= 20:
            return False
        # Only check for collisions if the piece is within the visible grid
        if y >= 0:
            if pos not in accepted_pos:
                # Treat garbage lines as solid blocks
                if y < 20 and grid[y][x] == (128, 128, 128):  # If we hit a garbage block
                    return False
                return False
    return True

def check_lost(positions):
    for pos in positions:
        x,y = pos
        # if y-coordinates less than 1 (0 or negative) --> the block is at top of play screen which means lost
        if y < 1:
            return True

    return False

def get_shape_from_index(index):
    piece = Piece(5, 0, shapes[index])  # Start at x=5 for better horizontal centering
    # Rotate I piece to be horizontal by default
    if index == 2:  # I piece
        piece.rotation = 1
    return piece

def hard_drop(piece, grid):
    # Move piece down until it can't move anymore
    while valid_space(piece, grid):
        piece.y += 1
    piece.y -= 1  # Move back up one step
    return True

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
        # pygame.draw.line(surface,color,start_pos,end_pos)
        pygame.draw.line(surface,(128,128,128),(top_left_x ,top_left_y + i*block_size),(top_left_x+play_width , top_left_y+ i * block_size))
        for j in range(len(grid_1[i])):
            pygame.draw.line(surface, (128, 128, 128), (top_left_x + j*block_size, top_left_y),(top_left_x  + j*block_size, top_left_y + play_height))

    for i in range(len(grid_2)):
        # pygame.draw.line(surface,color,start_pos,end_pos)
        pygame.draw.line(surface,(128,128,128),(top_left_x + add,top_left_y + i*block_size),(top_left_x+play_width + add, top_left_y+ i * block_size))
        for j in range(len(grid_2[i])):
            pygame.draw.line(surface, (128, 128, 128), (top_left_x + add + j*block_size, top_left_y),(top_left_x + add + j*block_size, top_left_y + play_height))

def clear_rows(grid, locked_pos):
    # delete rows part
    inc = 0
    for i in range(len(grid)-1,-1,-1):
        row = grid[i]
        # to check whether there are blank(black) spaces in row
        if (0,0,0) not in row:
            inc += 1
            # ind --> current index been looped (locate index)
            ind = i
            for j in range(len(row)):
                # delete the cells in the row which fulfill the conditions of being deleted
                try:
                    del locked_pos[(j,i)]
                except:
                    continue

    # to shift the rows
    # need to add back rows to the top after delete the row
    if inc > 0:
        # the sorted convert the list of locked_position dictionary keys to a list and sorted the values based on the second value of dict keys(tuple) in descending order
        # [(1, 0), (2, 0), (1, 1), (0, 0), (2, 1), (0, 1)] --> [(0, 1), (2, 1), (1, 1), (1, 0), (2, 0), (0, 0)]
        # the locked pos(after deleted some cells form code on top) will be accessed from bottom to top
        for key in sorted(list(locked_pos),key= lambda x:x[1],reverse=True):
            x,y = key
            # to only shift down the rows that above the row that been deleted
            if y < ind:
                newKey = (x,y+inc)
                locked_pos[newKey] = locked_pos.pop(key)

    return inc

def add_garbage_line(locked_positions, num_lines=1):
    # Create a new dictionary for the updated positions
    new_positions = {}
    
    # First, shift all existing blocks up by num_lines
    for y in range(19, -1, -1):
        for x in range(10):
            if (x, y) in locked_positions:
                new_y = y - num_lines  # Move blocks up by num_lines
                if new_y >= 0:  # Only keep blocks that don't go below the grid
                    new_positions[(x, new_y)] = locked_positions[(x, y)]
    
    # Then add garbage lines at the bottom
    for i in range(num_lines):
        gap_pos = random.randint(0, 9)
        for x in range(10):
            if x != gap_pos:
                new_positions[(x, 19 - i)] = (128, 128, 128)  # Gray color for garbage
    
    # Update the locked positions
    locked_positions.clear()
    locked_positions.update(new_positions)

def draw_next_pieces(pieces, surface, player_num):
    # Draw next pieces box
    next_x = top_left_x - 150 if player_num == 0 else top_left_x + play_width + 150
    next_y = top_left_y + 200
    
    # Draw box
    pygame.draw.rect(surface, (128, 128, 128), (next_x, next_y, 120, 360), 2)
    
    # Draw "NEXT" text
    font = pygame.font.SysFont('Consolas', 20, bold=True)
    label = font.render('NEXT', True, (255, 255, 255))
    surface.blit(label, (next_x + 40, next_y - 30))
    
    # Draw pieces
    for i, piece in enumerate(pieces):
        format = piece.shape[piece.rotation % len(piece.shape)]
        for y, line in enumerate(format):
            for x, column in enumerate(line):
                if column == '0':
                    pygame.draw.rect(surface, piece.color,
                                   (next_x + 30 + x*block_size,
                                    next_y + 30 + (i*120) + y*block_size,
                                    block_size, block_size), 0)

def draw_window(surface, grid_1, grid_2, current_piece_1, current_piece_2, score_1=0, score_2=0, level=1, speed=0.27, add=0):
    surface.fill((33,29,29))

    # initialize font object before creating it
    pygame.font.init()
    font = pygame.font.SysFont('Consolas',20,italic=True)
    label = font.render("LEVEL : "+str(level)+ "   SPEED: "+ str(round(1/speed,2)),True,(255,255,255))
    surface.blit(label, ((top_left_x + play_width) / 1.5 - label.get_width(), 30))
    surface.blit(label,((top_left_x+play_width)/1.5 - label.get_width() + add ,30))

    # draw the blocks
    # last arg represent border radius (0 = fill)
    for i in range(len(grid_1)):
        for j in range(len(grid_1[i])):
            pygame.draw.rect(surface, grid_1[i][j],
                             (top_left_x + (block_size * j), top_left_y + (block_size * i), block_size, block_size), 0)

    for i in range(len(grid_2)):
        for j in range(len(grid_2[i])):
            pygame.draw.rect(surface, grid_2[i][j],
                             (top_left_x + (block_size * j) + add, top_left_y + (block_size * i), block_size, block_size), 0)

    # Draw current pieces
    if current_piece_1:
        shape_pos = convert_shape_format(current_piece_1)
        for pos in shape_pos:
            x, y = pos
            if y > -1:  # Only draw if the piece is visible
                pygame.draw.rect(surface, current_piece_1.color,
                               (top_left_x + (block_size * x), top_left_y + (block_size * y), block_size, block_size), 0)

    if current_piece_2:
        shape_pos = convert_shape_format(current_piece_2)
        for pos in shape_pos:
            x, y = pos
            if y > -1:  # Only draw if the piece is visible
                pygame.draw.rect(surface, current_piece_2.color,
                               (top_left_x + (block_size * x) + add, top_left_y + (block_size * y), block_size, block_size), 0)

    # draw the border
    pygame.draw.rect(surface, (255, 0, 0), (top_left_x , top_left_y, play_width, play_height), 4)
    pygame.draw.rect(surface, (255, 0, 0), (top_left_x + add, top_left_y, play_width, play_height), 4)

    # draw the score
    font_1 = pygame.font.SysFont('Consolas', 20,bold=False,italic=True)
    label_1 = font_1.render('Score: '+ str(score_1), True, (255, 255, 255))

    font_2 = pygame.font.SysFont('Consolas', 20, bold=False, italic=True)
    label_2 = font_2.render('Score: ' + str(score_2), True, (255, 255, 255))

    x_coor = top_left_x + play_width + 50
    y_coor = top_left_y + play_height / 2 - 100

    # draw middle line
    pygame.draw.line(surface,(255,255,255),(s_width/2,0),(s_width/2,s_height))

    surface.blit(label_1, (x_coor + 10 , y_coor - 120))
    surface.blit(label_2,(x_coor+ 10 + add,y_coor-120))

    draw_grid(surface,grid_1,grid_2,add=int(mid_x))

    pygame.display.update()

def draw_hold_piece(piece, surface, player_num):
    # Draw hold piece box
    hold_x = top_left_x - 150 if player_num == 0 else top_left_x + play_width + 150
    hold_y = top_left_y + 50
    
    # Draw box
    pygame.draw.rect(surface, (128, 128, 128), (hold_x, hold_y, 120, 120), 2)
    
    # Draw "HOLD" text
    font = pygame.font.SysFont('Consolas', 20, bold=True)
    label = font.render('HOLD', True, (255, 255, 255))
    surface.blit(label, (hold_x + 40, hold_y - 30))
    
    if piece:  # Only draw if there's a piece to hold
        # Draw piece
        format = piece.shape[piece.rotation % len(piece.shape)]
        for i, line in enumerate(format):
            for j, column in enumerate(line):
                if column == '0':
                    pygame.draw.rect(surface, piece.color,
                                   (hold_x + 30 + j*block_size,
                                    hold_y + 30 + i*block_size,
                                    block_size, block_size), 0)

class Game:
    def __init__(self, surface):
        self.surface = surface
        self.clock = pygame.time.Clock()
        self.fall_time = 0
        self.level_time = 0
        self.level = 1
        self.fall_speed = 0.27
        
        # Initialize central block pool
        self.block_pool = BlockPool()
        
        # Initialize players
        self.player1 = Player(self.block_pool, True)
        self.player2 = Player(self.block_pool, False)
        
        # Initialize grids
        self.p1_grid = create_grid(self.player1.locked_positions)
        self.p2_grid = create_grid(self.player2.locked_positions)
    
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
        # Update timing
        self.fall_time += self.clock.get_rawtime()
        self.level_time += self.clock.get_rawtime()
        self.clock.tick()
        
        # Update level
        if self.level_time / 1000 > 15:
            self.level_time = 0
            if self.fall_speed > 0.12:
                self.fall_speed -= 0.01
                self.level += 1
        
        # Update piece positions
        if self.fall_time/1000 >= self.fall_speed:
            self.fall_time = 0
            self.player1.current_piece.y += 1
            self.player2.current_piece.y += 1
            
            if not(valid_space(self.player1.current_piece, self.p1_grid)) and self.player1.current_piece.y > 0:
                self.player1.current_piece.y -= 1
                self.player1.change_piece = True

            if not(valid_space(self.player2.current_piece, self.p2_grid)) and self.player2.current_piece.y > 0:
                self.player2.current_piece.y -= 1
                self.player2.change_piece = True
        
        # Update players and handle garbage
        self.update_player(self.player1, self.player2)
        self.update_player(self.player2, self.player1)
        
        # Update grids
        self.p1_grid = create_grid(self.player1.locked_positions)
        self.p2_grid = create_grid(self.player2.locked_positions)
        
        # Check for game over
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
        self.surface.fill((33,29,29))
        
        # Draw main grids with current pieces
        draw_window(self.surface, self.p1_grid, self.p2_grid, 
                   self.player1.current_piece, self.player2.current_piece,
                   self.player1.score, self.player2.score, 
                   self.level, self.fall_speed, add=int(mid_x))
        
        # Draw next pieces
        draw_next_pieces(self.player1.next_pieces, self.surface, 0)
        draw_next_pieces(self.player2.next_pieces, self.surface, 1)
        
        # Always draw hold pieces
        draw_hold_piece(self.player1.hold_piece, self.surface, 0)
        draw_hold_piece(self.player2.hold_piece, self.surface, 1)
        
        pygame.display.update()

def main(surface):
    game = Game(surface)
    run = True
    
    while run:
        for event in pygame.event.get():
            run = game.handle_input(event)
        
        run = game.update()
        game.draw()

def main_menu():
    run = True
    while run:
        win.fill((255, 178, 102))
        win.blit(background,(0,-70))
        font = pygame.font.SysFont('Consolas', 40,bold=True,italic=True)
        label = font.render('T E T R I S', True, (255, 255, 255))
        win.blit(label,(mid_x - label.get_width()/2, mid_y - label.get_height()/2 - 250))
        draw_text_bottom("PRESS ANY KEY TO START THE GAME",40,(255,255,255),win)

        small_font = pygame.font.SysFont('Consolas', 20, bold=True, italic=True)
        small_label = small_font.render('( PRESS I FOR INFO )', True, (255, 255, 255))
        win.blit(small_label,(mid_x - small_label.get_width()/2,mid_y + small_label.get_height()/2 + 310))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    info_page(win)
                else:
                    win.fill((0,0,0))
                    wait_font = pygame.font.SysFont('Consolas', 40,bold=True,italic=True)
                    wait_text = wait_font.render('STARTING GAME......', True, (255, 255, 255))
                    win.blit(wait_text,(mid_x-wait_text.get_width()/2,mid_y))
                    pygame.display.update()
                    pygame.time.delay(1500)
                    main(win)

    pygame.display.quit()

# Initialize pygame and start the game
win = pygame.display.set_mode((s_width,s_height))
pygame.display.set_caption("Tetris Game")
background = pygame.image.load('bgp.jpg')
main_menu()  # start game