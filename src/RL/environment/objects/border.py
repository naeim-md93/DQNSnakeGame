

class Border:
    def __init__(self, board_size):
        self.coords = self.get_coords(board_size=board_size)

    def get_coords(self, board_size):
        left_border = [(i, 0) for i in range(board_size[0])]
        right_border = [(i, board_size[1] - 1) for i in range(board_size[0])]
        up_border = [(0, i) for i in range(1, board_size[1] - 1)]
        down_border = [(board_size[0] - 1, i) for i in range(1, board_size[1] - 1)]
        return left_border + right_border + up_border + down_border