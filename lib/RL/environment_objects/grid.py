import pygame
import env


class Grid:
    def __init__(self):
        self.color = env.GRID_COLOR

    def draw(self, display_surface: pygame.Surface):
        """
        Draw grid on pygame display surface
        :param display_surface: pygame display surface
        :param display_size: display size
        :return: None
        """

        for i, l in enumerate(range(0, env.DISPLAY_SIZE[0], env.CELL_SIZE)):
            pygame.draw.line(
                surface=display_surface,
                color=self.color,
                start_pos=(0, l + i),
                end_pos=(env.DISPLAY_SIZE[1], l + i)
            )
        for i, t in enumerate(range(0, env.DISPLAY_SIZE[1], env.CELL_SIZE)):
            pygame.draw.line(
                surface=display_surface,
                color=self.color,
                start_pos=(t + i, 0),
                end_pos=(t + i, env.DISPLAY_SIZE[0])
            )
