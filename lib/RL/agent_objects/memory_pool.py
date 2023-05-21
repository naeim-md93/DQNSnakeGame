from collections import deque

import env


class MemoryPool:
    def __init__(self, size: int,
            proportions: list[float, float, float, float],
    ):
        self.NMP1 = deque(maxlen=int(size * proportions[0]))
        self.NMP2 = deque(maxlen=int(size * proportions[1]))
        self.PMP2 = deque(maxlen=int(size * proportions[2]))
        self.PMP1 = deque(maxlen=int(size * proportions[3]))

    def store_in_NMP1(self, xp):
        self.NMP1.append(xp)

    def store_in_NMP2(self, xp):
        self.NMP2.append(xp)

    def store_in_PMP2(self, xp):
        self.PMP2.append(xp)

    def store_in_PMP1(self, xp):
        self.PMP1.append(xp)