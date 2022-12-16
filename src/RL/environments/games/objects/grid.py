import pygame


class Grid:
    def __init__(self, color):
        self.color = color

    def draw(self, display_surface, cell_size, display_size):
        for l in range(0, display_size[0], cell_size[0]):
            pygame.draw.line(surface=display_surface, color=self.color, start_pos=(0, l), end_pos=(display_size[1], l))
        for t in range(0, display_size[1], cell_size[1]):
            pygame.draw.line(surface=display_surface, color=self.color, start_pos=(t, 0), end_pos=(t, display_size[0]))