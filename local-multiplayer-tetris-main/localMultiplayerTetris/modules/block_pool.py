import random
from .constants import shapes

class BlockPool:
    def __init__(self):
        self.pool = []
        self.refill_pool()

    def refill_pool(self):
        # Create a new pool with all shapes
        self.pool = list(range(len(shapes)))
        random.shuffle(self.pool)

    def get_next_piece(self):
        if not self.pool:
            self.refill_pool()
        return self.pool.pop(0) 