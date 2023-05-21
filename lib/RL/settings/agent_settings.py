"""
Value for specifying an object in 2d board
RGB color codes used by pygame while user is playing
and tensorboard to keep track of agent game play
"""
# Environment Settings ################
SEED = 42
TOTAL_GAMES = 15000
PYGAME = False
DEBUG = False
BOARD_SIZE = [14, 14]
CELL_SIZE = 20
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

# Exploration/Exploitation trade-off ##
EXPLORE_GAMES = 2000
INIT_EPSILON = 0.5
FINAL_EPSILON = 0.0
#######################################

# Important/Un-important memory trade-off
TRAIN_GAMES = 5000
INIT_ETA = 0.8
FINAL_ETA = 0.5
#######################################

# Memory pool settings ################
MAX_MEMORY = 2000000
MEMORY_THRESHOLD = 0.8
NMP1_PROPORTION = 0.125
NMP2_PROPORTION = 0.375
PMP2_PROPORTION = 0.375
PMP1_PROPORTION = 0.125
#######################################

# Train settings ######################
LEARNING_RATE = 0.0001
SHORT_MEMORY_GAMMA = 0.1
LONG_MEMORY_GAMMA = 0.9
BATCH_SIZE = 128
#######################################
