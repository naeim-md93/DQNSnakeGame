import numpy as np
from src.RL.environment.objects.grid import Grid
from src.RL.environment.objects.border import Border
from src.RL.environment.objects.obstacle import Obstacle


class Board:
    def __init__(
            self,
            board_size,
            board_directions,
            objects_values,
            snake_actions,
            max_obstacles
    ):
        self.board_size = board_size
        self.board_directions = board_directions
        self.objects_values = objects_values
        self.occupied_coords = []

        self.grid = Grid()
        
        self.border = Border(board_size=self.board_size)
        self.occupied_coords += self.border.coords
        
        self.obstacle = Obstacle(
            max_obstacles=max_obstacles,
            empty_coords=self.get_empty_coords(),
        )
        self.occupied_coords += self.obstacle.coords
    
    
    def get_empty_coords(self):
        coords = [(i, j) for i in range(self.board_size[0]) for j in range(self.board_size[1]) if (i, j) not in self.get_occupied_coords()]
        return coords
    
    def get_occupied_coords(self):
        return self.occupied_coords
    
    def get_board(self):
        board = np.ones(shape=self.board_size) * self.objects_values['BACKGROUND']
        
        for c in self.border.coords:
            board[c[0], c[1]] = self.objects_values['BORDER']
        
        for c in self.obstacle.coords:
            board[c[0], c[1]] = self.objects_values['OBSTACLE']
            
        return board
    
    def __repr__(self):
        return np.array2string(a=self.get_board())
        
        