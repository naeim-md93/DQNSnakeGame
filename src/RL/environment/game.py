import pygame
from src.RL.environment.objects.board import Board

class SnakeGame:
    def __init__(self, args):
        self.board_size = args.board_size
        self.cell_size = args.pygame_cell_size
        self.display_size = args.pygame_display_size
        self.objects_colors = args.env_objects_colors
        
        self.board = Board(
            board_size=self.board_size,
            board_directions=args.board_directions,
            objects_values=args.env_objects_values,
            snake_actions=args.snake_actions,
            max_obstacles=args.max_obstacles
        )
        
        if args.use_pygame:
            pygame.init()
            self.display_surface = pygame.display.set_mode(size=(self.display_size[1], self.display_size[0]))
            pygame.display.set_caption(args.pygame_game_name)
            self.clock = pygame.time.Clock()
            
            self.display()
        
    def display(self):        
        # Draw board
        self.display_surface.fill(color=self.objects_colors['BACKGROUND'])
        
        # Draw grid
        for y in range(0, self.display_size[0], self.cell_size[0] + 1):
            pygame.draw.line(
                surface=self.display_surface,
                color=self.objects_colors['GRID'],
                start_pos=(0, y),
                end_pos=(self.display_size[1], y)
            )
        for x in range(0, self.display_size[1], self.cell_size[1] + 1):
            pygame.draw.line(
                surface=self.display_surface,
                color=self.objects_colors['GRID'],
                start_pos=(x, 0),
                end_pos=(x, self.display_size[0])
            )
            
        # Draw border
        for c in self.board.border.coords:            
            l = (c[1] * self.cell_size[1]) + (c[1] + 1)
            t = (c[0] * self.cell_size[0]) + (c[0] + 1)
            border_rect = pygame.Rect(l, t, self.cell_size[1], self.cell_size[0])  # (left, top, width, height)
            pygame.draw.rect(surface=self.display_surface, color=self.objects_colors['BORDER'], rect=border_rect)

        # Draw obstacles
        for c in self.board.obstacle.coords:
            l = (c[1] * self.cell_size[1]) + (c[1] + 1)
            t = (c[0] * self.cell_size[0]) + (c[0] + 1)
            border_rect = pygame.Rect(l, t, self.cell_size[1], self.cell_size[0])  # (left, top, width, height)
            pygame.draw.rect(surface=self.display_surface, color=self.objects_colors['OBSTACLE'], rect=border_rect)
      
        
        # self.display_surface.blit(pygame.transform.rotate(surface=self.display_surface, angle=90), (0, 0))
        # self.display_surface.blit(pygame.transform.flip(surface=self.display_surface, flip_x=False, flip_y=True), (0, 0))
        pygame.display.update()
        pygame.event.pump()
        
