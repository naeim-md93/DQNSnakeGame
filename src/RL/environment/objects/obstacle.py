import random
from src.RL.environment.utils import get_8_neighbors


class Obstacle:
    def __init__(self, max_obstacles, empty_coords):
        self.max_obstacles = max_obstacles
        self.coords = self.get_coords(empty_coords=empty_coords)
        
    def get_coords(self, empty_coords):
        coords = []
        choices = empty_coords
        
        for i in range(self.max_obstacles):
            
            coord, choices = self.get_one_obstacle(choices=choices, empty_coords=empty_coords)
            
            if coord is not None:    
                coords.append(coord)
                empty_coords = [x for x in empty_coords if x != coord]
                
        return coords
    
    def get_one_obstacle(self, choices, empty_coords):
        valid_coord = None
        
        while len(choices) > 0:
            candidate = random.choice(seq=choices)
            
            choices = [x for x in choices if x != candidate]
            
            neighbors = get_8_neighbors(c=candidate)
            
            empty_neighbors = [x for x in neighbors if x in empty_coords]
            
            if len(empty_neighbors) == 8:
                valid_coord = candidate
                break
                
        return valid_coord, choices 

