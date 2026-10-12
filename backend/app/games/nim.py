class Nim:
    """Normal play: the player taking the final object wins."""
    def __init__(self,piles=(3,4,5),turn=1): self.piles=tuple(int(x) for x in piles); self.turn=turn
    def legal_moves(self): return [(i,n) for i,p in enumerate(self.piles) for n in range(1,p+1)]
    def play(self,move):
        i,n=move
        if move not in self.legal_moves(): raise ValueError('Illegal Nim move')
        p=list(self.piles);p[i]-=n;self.piles=tuple(p);self.turn*=-1
    def terminal(self): return all(p==0 for p in self.piles)
    def winner(self): return -self.turn if self.terminal() else 0
    def clone(self):return Nim(self.piles,self.turn)
    def key(self):return (self.piles,self.turn)
    def xor_move(self):
        x=0
        for p in self.piles:x^=p
        if not x:return None
        for i,p in enumerate(self.piles):
            target=p^x
            if target<p and (i,p-target) in self.legal_moves():return (i,p-target)
        return None

