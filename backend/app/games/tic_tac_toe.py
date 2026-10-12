class TicTacToe:
    def __init__(self, board=None, turn=1): self.board=list(board) if board is not None else [0]*9; self.turn=turn
    def legal_moves(self): return [i for i,v in enumerate(self.board) if v==0]
    def winner(self):
        for a,b,c in ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)):
            if self.board[a] and self.board[a]==self.board[b]==self.board[c]: return self.board[a]
        return 0
    def terminal(self): return self.winner()!=0 or not self.legal_moves()
    def play(self,move):
        if move not in self.legal_moves(): raise ValueError('Illegal move')
        self.board[move]=self.turn; self.turn*=-1
    def clone(self): return TicTacToe(self.board,self.turn)
    def key(self): return (tuple(self.board),self.turn)
