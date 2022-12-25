
WHITE = [255, 255, 255]  # Background
DARK_GREEN = [0, 155, 0]  # Snake Body
GREEN = [0, 255, 0]  # Snake Head
RED = [255, 0, 0]  # Apple
BLACK = [0, 0, 0]  # Borders
DARK_GRAY = [100, 100, 100]  # Grid


# Directions Index
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

SNAKE_ACTIONS = {'LEFT': LEFT, 'UP': UP, 'RIGHT': RIGHT, 'DOWN': DOWN}

BOARD_DIRECTIONS = {
    LEFT: {'ADD_TO_COORDS': (0, -1), 'INVALID': RIGHT},  # (+h, +w, index)
    UP: {'ADD_TO_COORDS': (-1, 0), 'INVALID': DOWN},  # (+h, +w, index)
    RIGHT: {'ADD_TO_COORDS': (0, 1), 'INVALID': LEFT},   # (+h, +w, index)
    DOWN: {'ADD_TO_COORDS': (1, 0), 'INVALID': UP}    # (+h, +w, index)
}

