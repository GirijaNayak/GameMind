from .minimax import minimax, SearchResult

def centre_first(state,moves):
    if getattr(state,'COLS',None): return sorted(moves,key=lambda c:abs(c-(state.COLS-1)/2))
    return moves
