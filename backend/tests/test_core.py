from app.main import search as search_endpoint, SearchRequest
from app.games.connect_four import ConnectFour
from app.games.tic_tac_toe import TicTacToe
from app.games.nim import Nim
from app.search.minimax import minimax
from app.search import centre_first
from app.evaluators.heuristic import handcrafted
import random

def test_connect_four_win_all_directions():
 boards=[]
 h=[[0]*7 for _ in range(6)];h[5][1:5]=[1]*4;boards.append(h)
 v=[[0]*7 for _ in range(6)]
 for r in range(2,6):v[r][3]=1
 boards.append(v)
 d1=[[0]*7 for _ in range(6)]
 for r,c in ((5,0),(4,1),(3,2),(2,3)):d1[r][c]=1
 boards.append(d1)
 d2=[[0]*7 for _ in range(6)]
 for r,c in ((2,0),(3,1),(4,2),(5,3)):d2[r][c]=1
 boards.append(d2)
 for b in boards:assert ConnectFour(b).winner()==1

def test_full_column_and_draw():
 s=ConnectFour()
 for _ in range(6):s.play(0)
 try:s.play(0);assert False
 except ValueError:pass
 b=[[1,-1,-1,-1,1,1,1],[-1,-1,1,1,-1,1,-1],[1,1,-1,-1,-1,1,-1],[-1,-1,1,1,1,-1,1],[1,-1,1,-1,-1,1,1],[1,1,-1,1,1,1,-1]]
 s=ConnectFour(b);assert s.terminal() and not s.winner()

def test_tictactoe_terminal_and_optimal():
 s=TicTacToe([1,1,0,-1,-1,0,0,0,0],1)
 assert minimax(s,9,lambda _:0).move==2
 s.play(2);assert s.winner()==1 and s.terminal()

def test_nim_xor_matches_minimax():
 s=Nim((1,2),1)
 m=s.xor_move();assert m is not None
 exact=minimax(s,sum(s.piles),lambda _:0,alpha_beta=True)
 assert exact.move==m
 s.play(m);assert Nim((1,2),1).terminal() is False

def test_search_alpha_beta_matches_plain():
 s=ConnectFour();a=minimax(s,2,handcrafted,False,centre_first);b=minimax(s,2,handcrafted,True,centre_first)
 assert a.value==b.value and a.move==b.move

def test_node_budget_and_api_search():
 s=ConnectFour();limited=minimax(s,6,handcrafted,move_order=centre_first,node_limit=3)
 assert limited.nodes<=3 and limited.move is not None

def test_node_limit_exact_iteration_boundaries():
 state=ConnectFour()
 exact=minimax(state,1,handcrafted,move_order=centre_first,node_limit=7)
 partial=minimax(state,1,handcrafted,move_order=centre_first,node_limit=6)
 assert exact.nodes==7 and exact.completed_depth==1 and len(exact.candidate_scores)==7
 assert partial.nodes==6 and partial.completed_depth==0 and len(partial.candidate_scores)==0
 assert minimax(state,4,handcrafted,move_order=centre_first,node_limit=1).nodes==1

def test_node_limit_exact_iteration_boundaries():
 state=ConnectFour()
 exact=minimax(state,1,handcrafted,move_order=centre_first,node_limit=7)
 partial=minimax(state,1,handcrafted,move_order=centre_first,node_limit=6)
 assert exact.nodes==7 and exact.completed_depth==1 and len(exact.candidate_scores)==7
 assert partial.nodes==6 and partial.completed_depth==0 and len(partial.candidate_scores)==0
 assert minimax(state,4,handcrafted,move_order=centre_first,node_limit=1).nodes==1

def test_connect_four_legal_moves_gravity_and_immutable_search():
 s=ConnectFour();assert s.legal_moves()==list(range(7))
 row=s.play(3)
 assert row==5 and s.turn==-1 and s.board[5][3]==1
 clone=s.clone();clone.play(3)
 assert s.board[4][3]==0 and clone.board[4][3]==-1
 state=ConnectFour();snapshot=(state.key(),[r[:] for r in state.board])
 minimax(state,3,handcrafted,move_order=centre_first)
 assert (state.key(),state.board)==snapshot

def test_connect_four_immediate_win_and_mandatory_block():
 win=ConnectFour()
 for c in (0,6,1,6,2,5):win.play(c)
 assert minimax(win,1,handcrafted,move_order=centre_first).move==3
 block=ConnectFour()
 for c in (4,0,5,1,6,2):block.play(c)
 assert block.turn==1
 assert minimax(block,2,handcrafted,move_order=centre_first).move==3
 yellow_win=ConnectFour()
 for c in (6,0,6,1,5,2,5):yellow_win.play(c)
 assert yellow_win.turn==-1
 yellow_result=minimax(yellow_win,1,handcrafted,move_order=centre_first)
 assert yellow_result.move==3 and yellow_result.value==-99999.0
 yellow_block=ConnectFour()
 for c in (0,6,1,5,2):yellow_block.play(c)
 assert yellow_block.turn==-1
 assert minimax(yellow_block,2,handcrafted,move_order=centre_first).move==3
 yellow_win=ConnectFour()
 for c in (6,0,6,1,5,2,5):yellow_win.play(c)
 assert yellow_win.turn==-1
 yellow_result=minimax(yellow_win,1,handcrafted,move_order=centre_first)
 assert yellow_result.move==3 and yellow_result.value==-99999.0
 yellow_block=ConnectFour()
 for c in (0,6,1,5,2):yellow_block.play(c)
 assert yellow_block.turn==-1
 assert minimax(yellow_block,2,handcrafted,move_order=centre_first).move==3

def test_alpha_beta_equivalence_on_seeded_reachable_positions():
 rng=random.Random(20261011)
 from app.evaluators.neural import NeuralEvaluator
 import json
 from app.paths import MODEL_DIR
 weights=json.loads((MODEL_DIR/'genetic_weights.json').read_text());neural=NeuralEvaluator(str(MODEL_DIR/'connect4.pt'))
 for position_index in range(6):
  state=ConnectFour()
  for _ply in range(rng.randint(2,20)):
   if state.terminal():break
   state.play(rng.choice(state.legal_moves()))
  if state.terminal():continue
  for depth in (1,2):
   plain=minimax(state,depth,handcrafted,alpha_beta=False,move_order=centre_first)
   pruned=minimax(state,depth,handcrafted,alpha_beta=True,move_order=centre_first)
   assert pruned.value==plain.value
   assert pruned.move==plain.move
   assert pruned.nodes<=plain.nodes
  if position_index<3:
   evaluators=(handcrafted,lambda s:handcrafted(s,weights),lambda s:neural(s)*s.turn)
   for evaluator in evaluators:
    plain=minimax(state,1,evaluator,alpha_beta=False,move_order=centre_first)
    pruned=minimax(state,1,evaluator,alpha_beta=True,move_order=centre_first)
    assert pruned.value==plain.value and pruned.move==plain.move

def test_yellow_root_minimizes_red_perspective_values():
 state=ConnectFour();state.play(3)
 result=minimax(state,1,handcrafted,move_order=centre_first)
 assert result.value==min(result.candidate_scores.values())

def test_evaluators_are_distinct_and_each_runs_inside_search():
 import json
 from pathlib import Path
 from app.evaluators.neural import NeuralEvaluator
 from app.paths import MODEL_DIR
 weights=json.loads((MODEL_DIR/'genetic_weights.json').read_text())
 neural=NeuralEvaluator(str(MODEL_DIR/'connect4.pt'))
 state=ConnectFour();state.play(3);state.play(2);state.play(3)
 evaluators=[handcrafted,lambda s:handcrafted(s,weights),lambda s:neural(s)*s.turn]
 outputs=[e(state) for e in evaluators]
 assert len({round(v,5) for v in outputs})==3
 for evaluator in evaluators:
  calls=[]
  def counted(s):calls.append(s.key());return evaluator(s)
  result=minimax(state,1,counted,move_order=centre_first)
  assert result.move in state.legal_moves() and calls
 win=ConnectFour()
 for c in (0,6,1,6,2,5):win.play(c)
 terminal_values=[minimax(win,1,e,move_order=centre_first).value for e in evaluators]
 assert terminal_values==[99999.0]*3

def test_api_search_and_validation():
 out=search_endpoint(SearchRequest(game='connect4',depth=1))
 assert out['move'] in range(7) and out['nodes']>0
from app.experiments.tournament import run_tournament
from pathlib import Path
import pytest
from fastapi import HTTPException

def test_trained_neural_evaluator_is_callable():
 from app.evaluators.neural import NeuralEvaluator
 score=NeuralEvaluator()(ConnectFour())
 assert isinstance(score,float) and -100<=score<=100
 with pytest.raises(FileNotFoundError):NeuralEvaluator(str(Path('missing-checkpoint-for-test.pt')))

def test_neural_training_uses_group_split_and_reports_mean_baseline(tmp_path):
 from app.evaluators.neural import train_positions
 from app.evaluators.heuristic import _fixed_positions
 states=_fixed_positions();labels=[float(s.turn) for s in states];groups=list(range(len(states)))
 meta=train_positions(states,labels,output=tmp_path/'tiny.pt',epochs=2,seed=5,groups=groups,dataset_metadata={'games_generated':len(groups),'source_group_ids':groups,'positions_removed_by_filter':0})
 assert meta['unique_train_groups']+meta['unique_validation_groups']+meta['unique_test_groups']==len(states)
 assert meta['training_mean_baseline_validation_mse']>=0
 assert meta['training_mean_baseline_test_mse']>=0
 assert meta['split'].startswith('70/15/15 by source game')
 assert meta['dataset_counts']['positions_before_deduplication']==len(states)
 assert meta['dataset_counts']['positions_removed_by_filter']==0
 assert len(meta['checkpoint_sha256'])==64

def test_experiment_records_csv_and_json(tmp_path):
 summary=run_tournament(matches=1,depth=1,budget_nodes=20,seed=19,outdir=tmp_path)
 assert summary['records']==3
 assert all(pair['matches']==1 for pair in summary['aggregate'])
 assert len(list(Path(tmp_path).glob('matches-*.csv')))==1
 assert len(list(Path(tmp_path).glob('summary-*.json')))==1
 assert summary['aggregate'][0]['matches']==1
 assert summary['results'][0]['nodes_a']>=0 and 'opening_id' in summary['results'][0]
 assert summary['results'][0]['max_nodes_on_move_a']<=summary['budget']['node_limit_per_move']
 assert summary['record_schema_version']==2 and summary['record_status']=='current'
 assert len(summary['artifact_fingerprints']['neural_checkpoint_sha256'])==64
 raw={(r['evaluator_a'],r['evaluator_b']):{'wins_a':0,'wins_b':0,'draws':0,'matches':0} for r in summary['results']}
 for row in summary['results']:
  item=raw[(row['evaluator_a'],row['evaluator_b'])];item['matches']+=1
  item['draws' if row['winner']=='draw' else 'wins_'+row['winner']]+=1
 for pair in summary['aggregate']:
  assert {k:pair[k] for k in ('wins_a','wins_b','draws','matches')}==raw[(pair['evaluator_a'],pair['evaluator_b'])]
 assert summary['outcome_ordering']['aggregate']=='wins_a-draws-wins_b'

def test_tournament_pairs_first_player_and_opening(tmp_path):
 summary=run_tournament(matches=2,depth=1,seed=23,outdir=tmp_path)
 for pair in summary['aggregate']:
  assert pair['a_first']==1 and pair['b_first']==1
 rows=summary['results']
 assert all(rows[i]['opening_id']==rows[i+1]['opening_id'] for i in range(0,len(rows),2))

def test_genetic_tournament_is_separate_from_proxy_fitness():
 from app.training.train_genetic import heldout_tournament
 from app.evaluators.heuristic import DEFAULT_WEIGHTS
 actual=heldout_tournament(DEFAULT_WEIGHTS,matches_per_position=2,depth=1)
 assert actual['matches']==8
 assert actual['wins']+actual['draws']+actual['losses']==8

def test_saved_model_metadata_matches_checkpoint_and_ga_weights():
 import hashlib,json
 from app.paths import MODEL_DIR
 neural_meta=json.loads((MODEL_DIR/'connect4.json').read_text())
 ga_meta=json.loads((MODEL_DIR/'genetic_experiment.json').read_text())
 assert 'realized terminal outcomes' in neural_meta['labels']
 assert 'not exact game-theoretic values' in neural_meta['labels']
 assert neural_meta['checkpoint_sha256']==hashlib.sha256((MODEL_DIR/'connect4.pt').read_bytes()).hexdigest()
 assert ga_meta['artifact_metadata']['weights_sha256']==hashlib.sha256((MODEL_DIR/'genetic_weights.json').read_bytes()).hexdigest()
 counts=neural_meta['dataset_counts']
 assert counts['positions_before_deduplication']==1545
 assert counts['positions_after_deduplication']==980
 assert sum(counts['games_by_split'].values())==89
from app.games.connect_four import ConnectFour
from app.main import SearchRequest, make_state

def test_api_accepts_gravity_state_and_rejects_floating_piece():
 state=ConnectFour();state.play(3)
 restored=make_state(SearchRequest(game='connect4',board=state.board,turn=state.turn,depth=1))
 assert restored.board==state.board and restored.turn==-1
 floating=[[0]*7 for _ in range(6)];floating[0][0]=1
 with pytest.raises(HTTPException):
  make_state(SearchRequest(game='connect4',board=floating,turn=-1))
def test_http_routes_lifespan_schemas_validation_and_experiment(tmp_path,monkeypatch):
 from fastapi.testclient import TestClient
 import app.main as api
 import app.experiments.tournament as runner
 monkeypatch.setattr(runner,'DATA_DIR',tmp_path)
 with TestClient(api.app) as client:
  health=client.get('/api/health')
  assert health.status_code==200 and health.json()['status']=='ok'
  connect=client.post('/api/search',json={'game':'connect4','depth':1,'evaluator':'handcrafted'})
  assert connect.status_code==200
  assert set(connect.json())=={'move','value','nodes','elapsed_ms','completed_depth','candidate_scores','pruned','evaluator','game'}
  assert connect.json()['game']=='connect4' and connect.json()['move'] in range(7)
  nim=client.post('/api/search',json={'game':'nim','piles':[1,2],'evaluator':'optimal'})
  assert nim.status_code==200 and nim.json()['game']=='nim'
  invalid=client.post('/api/search',json={'game':'connect4','board':[[0]],'depth':1})
  assert invalid.status_code==422
  experiment=client.post('/api/experiments/tournament',json={'matches':1,'depth':1,'budget':'nodes','budget_value':1,'seed':29})
  assert experiment.status_code==200 and experiment.json()['records']==3
  assert len(list(tmp_path.glob('summary-*.json')))==1
  history=client.get('/api/experiments/history')
  assert history.status_code==200
  assert all(r['record_status']=='legacy_unfingerprinted' for r in history.json() if r['record_schema_version']==1)

def test_http_neural_request_reports_missing_checkpoint_503(tmp_path,monkeypatch):
 from fastapi.testclient import TestClient
 import app.main as api
 monkeypatch.setattr(api,'MODEL_DIR',tmp_path)
 api._neural.cache_clear()
 try:
  with TestClient(api.app) as client:
   response=client.post('/api/search',json={'game':'connect4','depth':1,'evaluator':'neural'})
  assert response.status_code==503
  assert 'checkpoint' in response.json()['detail'].lower()
 finally:
  api._neural.cache_clear()
