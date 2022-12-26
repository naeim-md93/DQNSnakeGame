import CONSTANTS


def get_4_neighbors(c):
    return [(c[0] + v['ADD_TO_COORDS'][0], c[1] + v['ADD_TO_COORDS'][1]) for k, v in CONSTANTS.BOARD_DIRECTIONS.items()]


def get_8_neighbors(c):
    neighbors = []
    
    for k, v in CONSTANTS.BOARD_DIRECTIONS.items():

        n = (c[0] + v['ADD_TO_COORDS'][0], c[1] + v['ADD_TO_COORDS'][1])
        neighbors.append(n)
        
        if k == CONSTANTS.DIRECTION_VALUES['LEFT']:
            top_left = (
                n[0] + CONSTANTS.BOARD_DIRECTIONS[CONSTANTS.DIRECTION_VALUES['UP']]['ADD_TO_COORDS'][0],
                n[1] + CONSTANTS.BOARD_DIRECTIONS[CONSTANTS.DIRECTION_VALUES['UP']]['ADD_TO_COORDS'][1],
            )
            neighbors.append(top_left)
            
            bottom_left = (
                n[0] + CONSTANTS.BOARD_DIRECTIONS[CONSTANTS.DIRECTION_VALUES['DOWN']]['ADD_TO_COORDS'][0],
                n[1] + CONSTANTS.BOARD_DIRECTIONS[CONSTANTS.DIRECTION_VALUES['DOWN']]['ADD_TO_COORDS'][1],
            ) 
            neighbors.append(bottom_left)
        
        elif k == CONSTANTS.DIRECTION_VALUES['RIGHT']:
            top_right = (
                n[0] + CONSTANTS.BOARD_DIRECTIONS[CONSTANTS.DIRECTION_VALUES['UP']]['ADD_TO_COORDS'][0],
                n[1] + CONSTANTS.BOARD_DIRECTIONS[CONSTANTS.DIRECTION_VALUES['UP']]['ADD_TO_COORDS'][1],
            )
            neighbors.append(top_right)
            
            bottom_right = (
                n[0] + CONSTANTS.BOARD_DIRECTIONS[CONSTANTS.DIRECTION_VALUES['DOWN']]['ADD_TO_COORDS'][0],
                n[1] + CONSTANTS.BOARD_DIRECTIONS[CONSTANTS.DIRECTION_VALUES['DOWN']]['ADD_TO_COORDS'][1],
            ) 
            neighbors.append(bottom_right)
            
    return neighbors