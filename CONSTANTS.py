MAX_DISP_SIZE = [672, 1024]  # height, width

ENV_OBJECTS_COLORS = {
    'BACKGROUND': [255, 255, 255],  # White
    'GRID': [100, 100, 100],  # Dark Gray
    'BORDER': [0, 0, 0],  # Black
    'OBSTACLE': [0, 0, 0],  # Black
    'SNAKE_BODY': [0, 255, 0],  # Green
    'SNAKE_HEAD': [0, 155, 0],  # Dark Green
    'APPLE': [255, 0, 0],  # Red
}

ENV_OBJECTS_VALUES = {
    'BACKGROUND': 0,
    'BORDER': 1,
    'OBSTACLE': 1,
    'SNAKE_BODY': 2,
    'SNAKE_HEAD': 3,
    'APPLE': 4
}

DIRECTION_VALUES = {
    'LEFT': 0,
    'UP': 1,
    'RIGHT': 2,
    'DOWN': 3
}

SNAKE_ACTIONS = {
    'MOVE_LEFT': DIRECTION_VALUES['LEFT'],
    'MOVE_UP': DIRECTION_VALUES['UP'],
    'MOVE_RIGHT': DIRECTION_VALUES['RIGHT'],
    'MOVE_DOWN': DIRECTION_VALUES['DOWN'],

}

BOARD_DIRECTIONS = {
    DIRECTION_VALUES['LEFT']: {'ADD_TO_COORDS': (0, -1), 'INVALID': DIRECTION_VALUES['RIGHT']},
    DIRECTION_VALUES['UP']: {'ADD_TO_COORDS': (-1, 0), 'INVALID': DIRECTION_VALUES['DOWN']},
    DIRECTION_VALUES['RIGHT']: {'ADD_TO_COORDS': (0, 1), 'INVALID': DIRECTION_VALUES['LEFT']},
    DIRECTION_VALUES['DOWN']: {'ADD_TO_COORDS': (1, 0), 'INVALID': DIRECTION_VALUES['UP']},

}