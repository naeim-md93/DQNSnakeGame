import pygame
from src.RL.environment.objects.board import Board

class SnakeGame:
    def __init__(self, args):
        self.board_size = args.board_size
        self.cell_size = args.pygame_cell_size
        self.display_size = args.pygame_display_size
        
        self.board = Board(
            board_size=self.board_size,
            background_color=args.background_color,
            grid_color=args.grid_color
        )
        
        if args.use_pygame:
            pygame.init()
            self.display_surface = pygame.display.set_mode(size=(self.display_size[1], self.display_size[0]))
            pygame.display.set_caption(args.pygame_game_name)
            self.clock = pygame.time.Clock()
            
            self.display()
        
    def display(self):        
        # Draw board
        self.display_surface.fill(color=self.board.background_color)
        
        # Draw grid
        for y in range(0, self.display_size[0], self.cell_size[0] + 1):
            pygame.draw.line(surface=self.display_surface, color=self.board.grid.color, start_pos=(0, y), end_pos=(self.display_size[1], y))
        for x in range(0, self.display_size[1], self.cell_size[1] + 1):
            pygame.draw.line(surface=self.display_surface, color=self.board.grid.color, start_pos=(x, 0), end_pos=(x, self.display_size[0]))

        self.display_surface.blit(pygame.transform.rotate(surface=self.display_surface, angle=90), (0, 0))
        self.display_surface.blit(pygame.transform.flip(surface=self.display_surface, flip_x=False, flip_y=True), (0, 0))
        pygame.display.update()
        pygame.event.pump()
        
