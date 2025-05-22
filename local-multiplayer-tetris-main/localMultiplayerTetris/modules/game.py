import pygame
from .player import Player
from .constants import *

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((s_width, s_height))
        pygame.display.set_caption('Tetris')
        
        self.clock = pygame.time.Clock()
        self.fall_time = 0
        self.fall_speed = 0.5  # Time in seconds between piece movements
        
        # Initialize players
        self.player1 = Player(is_player_one=True)
        self.player2 = Player(is_player_one=False)
        
        # Game state
        self.game_over = False
        self.winner = None

    def run(self):
        while not self.game_over:
            self.fall_time += self.clock.get_rawtime()
            self.clock.tick()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    
                    # Handle player input
                    self.player1.handle_input(event)
                    self.player2.handle_input(event)

            # Update game state
            if self.fall_time / 1000 > self.fall_speed:
                self.fall_time = 0
                if not self.player1.action_handler.move_down():
                    if not self.player1.update():
                        self.game_over = True
                        self.winner = "Player 2"
                
                if not self.player2.action_handler.move_down():
                    if not self.player2.update():
                        self.game_over = True
                        self.winner = "Player 1"

            # Render
            self.draw()
            pygame.display.update()

        # Game over screen
        self.draw_game_over()
        pygame.display.update()
        pygame.time.wait(3000)  # Wait 3 seconds before closing

    def draw(self):
        self.screen.fill((0, 0, 0))
        
        # Draw player 1's grid
        self.draw_grid(self.player1, top_left_x, top_left_y)
        
        # Draw player 2's grid
        self.draw_grid(self.player2, mid_x + (s_width/2 - play_width) // 2, top_left_y)
        
        # Draw scores
        self.draw_score(self.player1, top_left_x - 150, top_left_y)
        self.draw_score(self.player2, mid_x + (s_width/2 - play_width) // 2 - 150, top_left_y)

    def draw_grid(self, player, x, y):
        # Draw the grid
        for i in range(len(player.grid)):
            for j in range(len(player.grid[i])):
                pygame.draw.rect(self.screen, player.grid[i][j],
                               (x + j * block_size, y + i * block_size, block_size, block_size), 0)
                pygame.draw.rect(self.screen, (128, 128, 128),
                               (x + j * block_size, y + i * block_size, block_size, block_size), 1)
        
        # Draw current piece
        if player.current_piece:
            for pos in player.current_piece.get_positions():
                pygame.draw.rect(self.screen, player.current_piece.color,
                               (x + pos[0] * block_size, y + pos[1] * block_size, block_size, block_size), 0)
        
        # Draw next pieces
        for i, piece_idx in enumerate(player.next_pieces):
            shape = shapes[piece_idx]
            for j, row in enumerate(shape[0]):
                for k, cell in enumerate(row):
                    if cell == '0':
                        pygame.draw.rect(self.screen, shape_colors[piece_idx],
                                       (x + play_width + 50 + k * block_size,
                                        y + i * 100 + j * block_size,
                                        block_size, block_size), 0)
        
        # Draw hold piece
        if player.hold_piece:
            shape = player.hold_piece.shape
            for j, row in enumerate(shape[0]):
                for k, cell in enumerate(row):
                    if cell == '0':
                        pygame.draw.rect(self.screen, player.hold_piece.color,
                                       (x - 150 + k * block_size,
                                        y + j * block_size,
                                        block_size, block_size), 0)

    def draw_score(self, player, x, y):
        font = pygame.font.SysFont('comicsans', 30)
        score_text = font.render(f'Score: {player.score}', True, (255, 255, 255))
        level_text = font.render(f'Level: {player.level}', True, (255, 255, 255))
        lines_text = font.render(f'Lines: {player.lines_cleared}', True, (255, 255, 255))
        
        self.screen.blit(score_text, (x, y))
        self.screen.blit(level_text, (x, y + 40))
        self.screen.blit(lines_text, (x, y + 80))

    def draw_game_over(self):
        font = pygame.font.SysFont('comicsans', 60)
        if self.winner:
            text = font.render(f'{self.winner} Wins!', True, (255, 255, 255))
        else:
            text = font.render('Game Over!', True, (255, 255, 255))
        
        text_rect = text.get_rect(center=(s_width/2, s_height/2))
        self.screen.blit(text, text_rect) 