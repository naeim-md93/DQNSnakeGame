from ..agent_objects.reward import distance_reward_function
"""
Value for specifying an object in 2d board
RGB color codes used by pygame while user is playing
and tensorboard to keep track of agent game play
"""
import math

# Environment Settings ################
SEED = 42
TOTAL_GAMES = 1
DEBUG = False
CELL_SIZE = 20
BOARD_SIZE = [14, 14]
#######################################

# Board settings ######################
BACKGROUND_COLOR = [255, 255, 255]  # WHITE
BACKGROUND_VALUE = 0
GRID_COLOR = [100, 100, 100]  # DARK_GRAY
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3
#######################################

# Border settings #####################
BORDER_COLOR = [0, 0, 0]  # BLACK
BORDER_VALUE = 1
#######################################

# Snake settings ######################
SNAKE_HEAD_COLOR = [0, 155, 0]  # DARK_GREEN
SNAKE_HEAD_VALUE = 3
SNAKE_BODY_COLOR = [0, 255, 0]  # GREEN
SNAKE_BODY_VALUE = 2
MIN_SNAKE_LENGTH = 1
#######################################

# Food settings ######################
FOOD_COLOR = [255, 0, 0]  # RED
FOOD_VALUE = 4
MIN_NUM_FOODS = 1
#######################################
