from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from .games import ConnectFour, TicTacToe, Nim
from .search import minimax, centre_first
from .evaluators.heuristic import handcrafted, DEFAULT_WEIGHTS
from .experiments.tournament import run_tournament
from .paths import DATA_DIR, MODEL_DIR

app=FastAPI(title='GameMind API',version='0.1.0',description='Game engines, search, evaluators, and reproducible experiments')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173'],allow_methods=['*'],allow_headers=['*'])
class SearchRequest(BaseModel):
 game: Literal['connect4','tictactoe','nim']
 board: list[list[int]]|None=None
 cells: list[int]|None=None
 piles: list[int]|None=None
 turn: int=Field(default=1,ge=-1,le=1)
 depth: int=Field(default=4,ge=1,le=12)
 evaluator: Literal['handcrafted','genetic','neural','optimal']='handcrafted'
 time_limit_ms: int|None=Field(default=None,ge=1,le=60000)
 alpha_beta: bool=True
 node_limit: int|None=Field(default=None,ge=1,le=10000000)
class TournamentRequest(BaseModel): matches: int=Field(20,ge=1,le=1000);depth: int=Field(3,ge=1,le=8);seed: int=13;budget: Literal['depth','time','nodes']='depth';budget_value:int|None=Field(default=None,ge=1,le=10000000)

def make_state(q):
 try:
  if q.turn not in (-1,1):raise ValueError('turn must be -1 or 1')
  if q.game=='connect4':
   board=q.board if q.board is not None else [[0]*7 for _ in range(6)]
   if len(board)!=6 or any(len(row)!=7 or any(v not in (-1,0,1) for v in row) for row in board):raise ValueError('Connect Four board must be 6x7 with cells -1, 0, or 1')
   for col in range(7):
    empty_seen=False
    for row in range(5,-1,-1):
     if board[row][col]==0:empty_seen=True
     elif empty_seen:raise ValueError('Connect Four pieces must obey gravity')
   red=sum(v==1 for row in board for v in row);yellow=sum(v==-1 for row in board for v in row)
   if red not in (yellow,yellow+1) or q.turn!=(1 if red==yellow else -1):raise ValueError('Connect Four piece counts do not match the side to move')
   return ConnectFour(board,q.turn)
  if q.game=='tictactoe':
   cells=q.cells or [0]*9
   if len(cells)!=9 or any(v not in (-1,0,1) for v in cells):raise ValueError('Tic-Tac-Toe cells must have length 9 and values -1, 0, or 1')
   red=cells.count(1);yellow=cells.count(-1)
   if red not in (yellow,yellow+1) or q.turn!=(1 if red==yellow else -1):raise ValueError('Tic-Tac-Toe piece counts do not match the side to move')
   return TicTacToe(cells,q.turn)
  if q.game=='nim':
   if q.piles is not None and (not q.piles or len(q.piles)>10 or any(v<0 or v>50 for v in q.piles)):raise ValueError('Nim piles must be 1 to 10 non-negative sizes, each at most 50')
   return Nim(q.piles or (3,4,5),q.turn)
 except (TypeError,ValueError): raise HTTPException(422,'Invalid game state')

from functools import lru_cache
@lru_cache(maxsize=2)
def _neural(path):
 from .evaluators.neural import NeuralEvaluator
 return NeuralEvaluator(path)

def eval_for(q,state):
 if q.game=='connect4':
  if q.evaluator=='neural':
   try:
    net=_neural(str(MODEL_DIR/'connect4.pt'));return lambda s:net(s)*s.turn
   except Exception as e:raise HTTPException(503,f'Neural evaluator unavailable: {e}')
  if q.evaluator=='genetic':
   from pathlib import Path
   p=MODEL_DIR/'genetic_weights.json'
   if p.exists():
    import json;weights=json.loads(p.read_text())
   else:raise HTTPException(503,'Genetic evaluator unavailable: train the model with python -m app.training.train_genetic')
   return lambda s:handcrafted(s,weights)
  return handcrafted
 if q.game=='tictactoe':
  # Exact terminal outcomes are propagated by search; favor centre/corners at cutoff.
  return lambda s:sum(v*(3 if i==4 else 2 if i in (0,2,6,8) else 1) for i,v in enumerate(s.board))
 if q.evaluator=='optimal' and isinstance(state,Nim):return lambda s: 1 if __import__('functools').reduce(int.__xor__,s.piles,0)!=0 else -1
 return lambda s:(1 if __import__('functools').reduce(int.__xor__,s.piles,0)!=0 else -1)*s.turn

@app.get('/api/health')
def health():return {'status':'ok','games':['connect4','tictactoe','nim'],'evaluators':['handcrafted','genetic','neural','optimal']}
@app.post('/api/search')
def search(q:SearchRequest):
 s=make_state(q)
 if s.terminal():raise HTTPException(400,'Game is already over')
 if q.game=='nim' and q.evaluator=='optimal':
  xor=__import__('functools').reduce(int.__xor__,s.piles,0)
  m=s.xor_move() or s.legal_moves()[0]
  predicted_winner=s.turn if xor else -s.turn
  return {'move':m,'value':predicted_winner,'nodes':1,'elapsed_ms':0.0,'completed_depth':0,'candidate_scores':{},'pruned':0,'evaluator':'optimal','game':'nim'}
 if q.game=='tictactoe' and s.winner():raise HTTPException(422,'Board is terminal')
 evaluator=eval_for(q,s)
 ordering=centre_first if q.game=='connect4' else None
 try:r=minimax(s,q.depth,evaluator,alpha_beta=q.alpha_beta,move_order=ordering,time_limit_ms=q.time_limit_ms,node_limit=q.node_limit)
 except Exception as e:raise HTTPException(400,str(e))
 return {'move':r.move,'value':r.value,'nodes':r.nodes,'elapsed_ms':round(r.elapsed_ms,3),'completed_depth':r.completed_depth,'candidate_scores':r.candidate_scores,'pruned':r.pruned,'evaluator':q.evaluator,'game':q.game}
@app.post('/api/experiments/tournament')
def tournament(q:TournamentRequest):
 if q.budget!='depth' and q.budget_value is None:raise HTTPException(422,'budget_value is required for node and time budgets')
 return run_tournament(q.matches,q.depth,budget_nodes=q.budget_value if q.budget=='nodes' else None,time_limit_ms=q.budget_value if q.budget=='time' else None,seed=q.seed)
@app.get('/api/experiments/history')
def history():
 from pathlib import Path
 import json
 files=sorted(DATA_DIR.glob('summary-*.json'),reverse=True)
 records=[]
 for p in files:
  record=json.loads(p.read_text())
  if record.get('record_schema_version')!=2:
   record.setdefault('record_status','legacy_unfingerprinted')
   record.setdefault('status_note','Historical result without current artifact fingerprints or per-move maxima.')
  records.append(record)
 return records


