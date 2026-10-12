from enum import Enum
from typing import Optional

class Player(int, Enum):
    RED = 1
    YELLOW = -1

class ConnectFour:
    ROWS, COLS = 6, 7
    def __init__(self, board=None, turn=1):
        self.board = [list(row) for row in board] if board is not None else [[0]*self.COLS for _ in range(self.ROWS)]
        self.turn = int(turn)
    def legal_moves(self): return [c for c in range(self.COLS) if self.board[0][c] == 0]
    def play(self, col):
        if col not in self.legal_moves(): raise ValueError('Column is full or out of range')
        for r in range(self.ROWS-1, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = self.turn; self.turn *= -1; return r
    def winner(self):
        for r in range(self.ROWS):
            for c in range(self.COLS):
                p=self.board[r][c]
                if not p: continue
                for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
                    cells=[(r+i*dr,c+i*dc) for i in range(4)]
                    if all(0<=rr<self.ROWS and 0<=cc<self.COLS and self.board[rr][cc]==p for rr,cc in cells): return p
        return 0
    def terminal(self): return self.winner()!=0 or not self.legal_moves()
    def winning_cells(self):
        w=self.winner()
        if not w:return []
        for r in range(self.ROWS):
            for c in range(self.COLS):
                for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
                    cells=[(r+i*dr,c+i*dc) for i in range(4)]
                    if all(0<=rr<self.ROWS and 0<=cc<self.COLS and self.board[rr][cc]==w for rr,cc in cells): return cells
        return []
    def clone(self): return ConnectFour(self.board,self.turn)
    def key(self): return (tuple(v for row in self.board for v in row),self.turn)
    def mirrored_key(self): return (tuple(v for row in self.board for v in row[::-1]),self.turn)
    def to_dict(self): return {'board':self.board,'turn':self.turn,'winner':self.winner(),'draw':self.winner()==0 and not self.legal_moves(),'legal_moves':self.legal_moves(),'winning_cells':self.winning_cells()}
