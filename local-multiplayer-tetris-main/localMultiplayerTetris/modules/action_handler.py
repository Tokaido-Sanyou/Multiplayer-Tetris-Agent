class ActionHandler:
    def __init__(self, player):
        self.player = player

    def move_left(self):
        if self.player.current_piece:
            self.player.current_piece.x -= 1
            if not self._is_valid_move():
                self.player.current_piece.x += 1

    def move_right(self):
        if self.player.current_piece:
            self.player.current_piece.x += 1
            if not self._is_valid_move():
                self.player.current_piece.x -= 1

    def move_down(self):
        if self.player.current_piece:
            self.player.current_piece.y += 1
            if not self._is_valid_move():
                self.player.current_piece.y -= 1
                self._lock_piece()
                return True
        return False

    def hard_drop(self):
        if self.player.current_piece:
            while self._is_valid_move():
                self.player.current_piece.y += 1
            self.player.current_piece.y -= 1
            self._lock_piece()
            return True
        return False

    def rotate_cw(self):
        if self.player.current_piece:
            return self.player.current_piece.rotate(self.player.grid, clockwise=True)
        return False

    def rotate_ccw(self):
        if self.player.current_piece:
            return self.player.current_piece.rotate(self.player.grid, clockwise=False)
        return False

    def hold_piece(self):
        if not self.player.can_hold:
            return False
        
        if self.player.hold_piece is None:
            self.player.hold_piece = self.player.current_piece
            self.player.current_piece = self.player.get_new_piece()
        else:
            self.player.current_piece, self.player.hold_piece = self.player.hold_piece, self.player.current_piece
            self.player.current_piece.x = 3
            self.player.current_piece.y = 0
        
        self.player.can_hold = False
        return True

    def _is_valid_move(self):
        if not self.player.current_piece:
            return False
        
        for x, y in self.player.current_piece.get_positions():
            if not (0 <= x < len(self.player.grid[0]) and 
                   0 <= y < len(self.player.grid) and 
                   self.player.grid[y][x] == (0, 0, 0)):
                return False
        return True

    def _lock_piece(self):
        if not self.player.current_piece:
            return
        
        for x, y in self.player.current_piece.get_positions():
            self.player.locked_positions[(x, y)] = self.player.current_piece.color
        
        self.player.current_piece = None
        self.player.can_hold = True 