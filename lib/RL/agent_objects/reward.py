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


def manhattan_distance(a, b):
    """
    Calculates the Manhattan distance between two points in 2D space.
    :param a: A tuple representing the first point (x1, y1).
    :param b: A tuple representing the second point (x2, y2).
    :return: The Euclidean distance between the two points.
    """
    x1, y1 = a
    x2, y2 = b
    return abs(x2 - x1) + abs(y2 - y1)


def get_closest_food(head_coords, foods):
    tmp = set()

    for f in foods:
        tmp.add((euclidean_distance(a=head_coords, b=f), f))

    return min(tmp, key=lambda x: x[0])[1]


def distance_reward_function(snake_coords, food_coords, choice, init_snake_length):

    if len(food_coords) > 1:
        food = get_closest_food(head_coords=choice, foods=food_coords)
    else:
        food = tuple(food_coords)[0]

    dis = manhattan_distance(a=choice, b=food)

    return (1 / (0.5 * dis + 1)) * (init_snake_length / len(snake_coords))


REWARDS = {
    'FOOD': lambda score, max_score: math.e ** (1 + score / max_score),
    'GAME_OVER': -math.e ** 2,
    'ELSE': distance_reward_function,
    'TIMEOUT': -1
}
