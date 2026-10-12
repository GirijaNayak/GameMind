import time
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class SearchResult:
    move: Any
    value: float
    nodes: int
    elapsed_ms: float
    completed_depth: int
    candidate_scores: dict
    pruned: int = 0

def minimax(state, depth, evaluator:Callable, alpha_beta=True, move_order=None, time_limit_ms=None, iterative=True, node_limit=None):
    start=time.perf_counter(); nodes=0; pruned=0; root_scores={}; best=None; best_value=float('-inf'); completed=0; stopped=False
    order=move_order or (lambda s,moves: moves)
    def visit(s,d,alpha,beta,ply):
        nonlocal nodes,pruned,stopped
        if node_limit is not None and nodes>=node_limit:
            stopped=True;return float(evaluator(s))
        if time_limit_ms is not None and (time.perf_counter()-start)*1000>=time_limit_ms:
            stopped=True;return float(evaluator(s))
        nodes+=1
        winner=getattr(s,'winner',lambda:0)()
        if winner: return (100000-ply)*winner
        if s.terminal(): return 0
        if d==0:return float(evaluator(s))
        moves=order(s,s.legal_moves()); maximizing=s.turn==1; value=float('-inf') if maximizing else float('inf')
        for idx,m in enumerate(moves):
            if stopped:break
            if node_limit is not None and nodes>=node_limit:
                stopped=True;break
            child=s.clone();child.play(m);score=visit(child,d-1,alpha,beta,ply+1)
            if maximizing:value=max(value,score);alpha=max(alpha,value)
            else:value=min(value,score);beta=min(beta,value)
            if alpha_beta and alpha>=beta:
                pruned+=len(moves)-idx-1;break
        if value in (float('-inf'),float('inf')):return float(evaluator(s))
        return value
    root_maximizing=state.turn==1
    for current_depth in range(1,depth+1) if iterative else [depth]:
        if (time_limit_ms is not None and (time.perf_counter()-start)*1000>=time_limit_ms) or (node_limit is not None and nodes>=node_limit):break
        iteration={}; local_best=None; local_val=float('-inf') if root_maximizing else float('inf');moves=order(state,state.legal_moves());finished=True
        for m in moves:
            if stopped or (node_limit is not None and nodes>=node_limit) or (time_limit_ms is not None and (time.perf_counter()-start)*1000>=time_limit_ms):finished=False;break
            child=state.clone();child.play(m);v=visit(child,current_depth-1,float('-inf'),float('inf'),1)
            if stopped:finished=False;break
            iteration[str(m)]=v
            if (root_maximizing and v>local_val) or (not root_maximizing and v<local_val):local_val=v;local_best=m
            if stopped or (time_limit_ms is not None and (time.perf_counter()-start)*1000>=time_limit_ms):finished=False;break
        if iteration and finished:
            root_scores=iteration;best=local_best;best_value=local_val;completed=current_depth
        if (time_limit_ms is not None and (time.perf_counter()-start)*1000>=time_limit_ms) or (node_limit is not None and nodes>=node_limit) or not moves:break
    if best is None:
        legal=order(state,state.legal_moves())
        if legal:best=legal[0];best_value=float(evaluator(state))
    return SearchResult(best,best_value,nodes,(time.perf_counter()-start)*1000,completed,root_scores,pruned)
