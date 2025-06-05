import pygame
import random
from constants import top_left_x, play_width, play_height, block_size, mid_x, mid_y, s_width, s_height, top_left_y, shapes
from piece_utils import convert_shape_format, valid_space

def create_grid(locked_positions={}):
    # create a blank (black) 2-d array grid
    grid = [[(0,0,0) for _ in range(10)] for _ in range(20)]

    # Update grid based on locked positions
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j,i) in locked_positions:
                c = locked_positions[(j,i)]
                grid[i][j] = c

    #print("Grid initialization:", grid[:5])  # Debug: Log the top 5 rows of the grid
    return grid

def check_lost(positions):
    for pos in positions:
        x, y = pos
        # only lose if a locked block is above row 0
        if y < 0:
            return True
    return False

def get_shape_from_index(index):
    from piece import Piece
    piece = Piece(5, -1, shapes[index])  # Spawns at y = -1
    if index == 2:  # I piece
        piece.rotation = 1
    return piece

def hard_drop(piece, grid):
    while valid_space(piece, grid):
        piece.y += 1
    piece.y -= 1
    return True

def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont("Consolas",size,bold=True)
    label = font.render(str(text),True,color)
    surface.blit(label,(top_left_x + play_width/2 - label.get_width()/2, 
                       top_left_y + play_height/2 - label.get_height()/2))

def draw_text_bottom(text, size, color, surface):
    font = pygame.font.SysFont("Consolas",size,bold=True,italic=True)
    label = font.render(str(text),True,color)
    surface.blit(label,(mid_x - label.get_width()/2, 
                       mid_y + label.get_height()/2 + 250))

def draw_grid(surface, grid_1, grid_2, add=0):
    for i in range(len(grid_1)):
        pygame.draw.line(surface,(128,128,128),
                        (top_left_x, top_left_y + i*block_size),
                        (top_left_x+play_width, top_left_y + i*block_size))
        for j in range(len(grid_1[i])):
            pygame.draw.line(surface, (128, 128, 128),
                           (top_left_x + j*block_size, top_left_y),
                           (top_left_x + j*block_size, top_left_y + play_height))

    for i in range(len(grid_2)):
        pygame.draw.line(surface,(128,128,128),
                        (top_left_x + add, top_left_y + i*block_size),
                        (top_left_x+play_width + add, top_left_y + i*block_size))
        for j in range(len(grid_2[i])):
            pygame.draw.line(surface, (128, 128, 128),
                           (top_left_x + add + j*block_size, top_left_y),
                           (top_left_x + add + j*block_size, top_left_y + play_height))

def clear_rows(grid, locked_pos):
    inc = 0
    for i in range(len(grid)-1,-1,-1):
        row = grid[i]
        if (0,0,0) not in row:
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del locked_pos[(j,i)]
                except:
                    continue

    if inc > 0:
        for key in sorted(list(locked_pos),key= lambda x:x[1],reverse=True):
            x,y = key
            if y < ind:
                newKey = (x,y+inc)
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

    # Draw current pieces directly at grid positions
    if current_piece_1:
        for x, y in convert_shape_format(current_piece_1):
            if 0 <= x < 10 and y >= 0:
                pygame.draw.rect(surface, current_piece_1.color,
                               (top_left_x + (block_size * x), top_left_y + (block_size * y), block_size, block_size), 0)
    if current_piece_2:
        for x, y in convert_shape_format(current_piece_2):
            if 0 <= x < 10 and y >= 0:
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

def count_holes(grid):
    """
    Count the number of holes in a Tetris grid.
    A hole is defined as an empty cell below at least one filled cell in the same column.
    grid: 2D list of color tuples
    Returns: integer count of holes
    """
    holes = 0
    width = len(grid[0])
    height = len(grid)
    for x in range(width):
        block_found = False
        for y in range(height):
            if grid[y][x] != (0, 0, 0):
                block_found = True
            elif block_found and grid[y][x] == (0, 0, 0):
                holes += 1
    return holes