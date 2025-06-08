import random

class BlockPool:
    def __init__(self):
        self.pool = []
        self.refill_pool()
    
    def refill_pool(self):
        # Create a new bag of all 7 pieces
        bag = list(range(7))  # 0-6 representing all pieces
        random.shuffle(bag)
        self.pool.extend(bag)
    
    def ensure_blocks_ahead(self, player_index):
        # Make sure there are at least 7 blocks ahead of the player's current position
        while len(self.pool) - player_index < 7:
            self.refill_pool()
    
    def get_block_at(self, index):
        return self.pool[index]
    
    def get_next_blocks(self, current_index, count=3):
        # Get the next 'count' blocks starting from current_index + 1
        return [self.pool[current_index + 1 + i] for i in range(count)] 