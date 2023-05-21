import math


def euclidean_distance(a, b):
    """
    Calculates the Euclidean distance between two points in 2D space.
    :param a: A tuple representing the first point (x1, y1).
    :param b: A tuple representing the second point (x2, y2).
    :return: The Euclidean distance between the two points.
    """
    x1, y1 = a
    x2, y2 = b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_closest_food(head_coords, foods):
    tmp = set()

    for f in foods:
        tmp.add((euclidean_distance(a=head_coords, b=f), f))

    return min(tmp, key=lambda x: x[0])[1]


def distance_reward_function(board_coords, snake_coords, food_coords, choice, init_snake_length):

    if len(food_coords) > 1:
        food = get_closest_food(head_coords=snake_coords[0], foods=food_coords)
    else:
        food = tuple(food_coords)[0]

    min_height = min(board_coords, key=lambda x: x[0])[0]
    max_height = max(board_coords, key=lambda x: x[0])[0]
    min_width = min(board_coords, key=lambda x: x[1])[1]
    max_width = max(board_coords, key=lambda x: x[1])[1]
    max_dis = euclidean_distance(a=(min_height, min_width), b=(max_height, max_width))
    old_dis = euclidean_distance(a=snake_coords[0], b=food)
    new_dis = euclidean_distance(a=choice, b=food)

    closer = old_dis > new_dis
    score = (max_dis - new_dis) / max_dis

    reward = score * closer + (score - 1) * (1 - closer)
    reward = reward / (len(snake_coords) - init_snake_length + 1)

    return reward


REWARDS = {
    'FOOD': lambda score, max_score: 5,
    'GAME_OVER': -2.5,
    'ELSE': distance_reward_function,
    'TIMEOUT': -1
}
