from src.RL.environment.objects.grid import Grid


class Board:
    def __init__(
            self,
            board_size,
            background_color,
            grid_color,
    ):
        self.board_size = board_size
        self.background_color = background_color

        self.grid = Grid(color=grid_color)
        
        