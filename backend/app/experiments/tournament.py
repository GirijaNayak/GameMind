import csv,json,random
import hashlib
from pathlib import Path
from datetime import datetime,timezone
from ..games.connect_four import ConnectFour
from ..search.minimax import minimax
from ..search import centre_first
from ..evaluators.heuristic import handcrafted
from ..paths import DATA_DIR, MODEL_DIR

# Reused openings are legal, nonterminal, and paired with both evaluator seat assignments.
OPENINGS=((),(3,2,3,4),(2,3,4),(3,3,2,4,2,5),(3,2,3))

def _sha256(path):
    if not path.is_file():return None
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def _play_opening(moves):
    state=ConnectFour()
    for move in moves:
        state.play(move)
    if state.terminal():raise ValueError(f'Opening is terminal: {moves}')
    return state

def run_tournament(matches=20,depth=3,budget_nodes=None,time_limit_ms=None,seed=13,outdir=None):
    """Run a configurable round robin; matches are games per evaluator pair."""
    if matches<1:raise ValueError('matches must be at least one')
    if depth<1:raise ValueError('depth must be at least one')
    if budget_nodes is not None and time_limit_ms is not None:raise ValueError('choose either a node budget or a time budget')
    agents={'handcrafted':handcrafted};pending=[]
    genetic_path=MODEL_DIR/'genetic_weights.json'
    if genetic_path.exists():
        weights=json.loads(genetic_path.read_text());agents['genetic']=lambda s:handcrafted(s,weights)
    else:pending.append('genetic evaluator tournament pending: run python -m app.training.train_genetic')
    neural_path=MODEL_DIR/'connect4.pt'
    if neural_path.exists():
        from ..evaluators.neural import NeuralEvaluator
        net=NeuralEvaluator(str(neural_path));agents['neural']=lambda s:net(s)*s.turn
    else:pending.append('neural evaluator tournament pending: run python -m app.training.train_connect4')
    rows=[];names=list(agents);opening_order=list(range(len(OPENINGS)));random.Random(seed).shuffle(opening_order)
    for i,a in enumerate(names):
      for b in names[i+1:]:
       for game in range(matches):
        pair_id=game//2;opening_id=opening_order[pair_id%len(OPENINGS)];opening=OPENINGS[opening_id]
        player_a=1 if game%2==0 else -1;state=_play_opening(opening);first_player='a' if player_a==state.turn else 'b'
        nodes_a=nodes_b=moves_a=moves_b=max_nodes_a=max_nodes_b=0;elapsed_a=elapsed_b=max_elapsed_a=max_elapsed_b=0.0;depth_a=depth_b=0
        while not state.terminal():
         who=a if state.turn==player_a else b
         result=minimax(state,depth,agents[who],move_order=centre_first,time_limit_ms=time_limit_ms,node_limit=budget_nodes)
         if result.move is None:break
         state.play(result.move)
         if who==a:
          nodes_a+=result.nodes;elapsed_a+=result.elapsed_ms;moves_a+=1;max_nodes_a=max(max_nodes_a,result.nodes);max_elapsed_a=max(max_elapsed_a,result.elapsed_ms);depth_a=max(depth_a,result.completed_depth)
         else:
          nodes_b+=result.nodes;elapsed_b+=result.elapsed_ms;moves_b+=1;max_nodes_b=max(max_nodes_b,result.nodes);max_elapsed_b=max(max_elapsed_b,result.elapsed_ms);depth_b=max(depth_b,result.completed_depth)
        winner=state.winner();rows.append({'record_schema_version':2,'match_id':f'{seed}-{a}-{b}-{game}','pair_id':f'{a}-{b}-{seed}-{pair_id}','opening_id':opening_id,'opening_moves':list(opening),'evaluator_a':a,'evaluator_b':b,'winner':'draw' if not winner else ('a' if winner==player_a else 'b'),'first_player':first_player,'moves_a':moves_a,'moves_b':moves_b,'nodes_a':nodes_a,'nodes_b':nodes_b,'max_nodes_on_move_a':max_nodes_a,'max_nodes_on_move_b':max_nodes_b,'elapsed_ms_a':round(elapsed_a,3),'elapsed_ms_b':round(elapsed_b,3),'max_elapsed_ms_on_move_a':round(max_elapsed_a,3),'max_elapsed_ms_on_move_b':round(max_elapsed_b,3),'completed_depth_a':depth_a,'completed_depth_b':depth_b,'seed':seed,'budget':'time' if time_limit_ms is not None else ('nodes' if budget_nodes is not None else 'depth'),'budget_value':time_limit_ms if time_limit_ms is not None else budget_nodes,'status':'recorded'})
    root=Path(outdir) if outdir else DATA_DIR;root.mkdir(parents=True,exist_ok=True);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');csvp=root/f'matches-{stamp}.csv';jsonp=root/f'summary-{stamp}.json'
    aggregate={}
    for row in rows:
      key=f"{row['evaluator_a']}__vs__{row['evaluator_b']}"
      item=aggregate.setdefault(key,{'evaluator_a':row['evaluator_a'],'evaluator_b':row['evaluator_b'],'wins_a':0,'wins_b':0,'draws':0,'matches':0,'a_first':0,'b_first':0})
      item['matches']+=1;item['a_first' if row['first_player']=='a' else 'b_first']+=1
      if row['winner']=='draw':item['draws']+=1
      elif row['winner']=='a':item['wins_a']+=1
      else:item['wins_b']+=1
    if rows:
      with csvp.open('w',newline='',encoding='utf-8') as f:
       w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    artifacts={'neural_checkpoint_sha256':_sha256(neural_path),'neural_metadata_sha256':_sha256(MODEL_DIR/'connect4.json'),'genetic_weights_sha256':_sha256(genetic_path),'genetic_experiment_sha256':_sha256(MODEL_DIR/'genetic_experiment.json')}
    summary={'record_schema_version':2,'record_status':'current','created_utc':stamp,'seed':seed,'opening_order':opening_order,'matches_requested_per_pair':matches,'records':len(rows),'evaluators_available':names,'aggregate':list(aggregate.values()),'outcome_ordering':{'aggregate':'wins_a-draws-wins_b','raw':'winner field is a, b, or draw'},'artifact_fingerprints':artifacts,'results':rows,'pending':pending,'limitations':'Exploratory deterministic matches. Evaluator A alternates first player in paired games from the same opening; repeated openings and finite samples do not establish general tournament strength. Per-move node/time budgets are reset for each move; totals below sum actual measurements across each whole game. Time limits are checked between search nodes and are soft, so wall time can exceed the requested limit.','budget':{'type':'time' if time_limit_ms is not None else ('nodes' if budget_nodes is not None else 'depth'),'node_limit_per_move':budget_nodes,'time_limit_ms_per_move':time_limit_ms,'depth_cap':depth,'opening_pool':len(OPENINGS)}}
    jsonp.write_text(json.dumps(summary,indent=2));return summary
