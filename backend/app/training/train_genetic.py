import json
import hashlib
from pathlib import Path
from ..evaluators.heuristic import genetic_optimize
from ..paths import MODEL_DIR
from ..evaluators.heuristic import handcrafted, DEFAULT_WEIGHTS, _fixed_positions
from ..search.minimax import minimax
from ..search import centre_first

def heldout_tournament(weights,matches_per_position=2,depth=3):
 """Play the evolved evaluator against default weights on untouched reachable states."""
 if matches_per_position<1:raise ValueError('matches_per_position must be at least one')
 heldout=_fixed_positions()[8:12];rows=[];wins=draws=losses=0
 for position_id,initial in enumerate(heldout):
  for game in range(matches_per_position):
   candidate_seat=1 if game%2==0 else -1;state=initial.clone();candidate_nodes=baseline_nodes=0
   while not state.terminal():
    is_candidate=state.turn==candidate_seat
    evaluator=(lambda s:handcrafted(s,weights)) if is_candidate else (lambda s:handcrafted(s,DEFAULT_WEIGHTS))
    result=minimax(state,depth,evaluator,move_order=centre_first)
    state.play(result.move)
    if is_candidate:candidate_nodes+=result.nodes
    else:baseline_nodes+=result.nodes
   winner=state.winner()*candidate_seat
   outcome='draw' if winner==0 else ('win' if winner==1 else 'loss')
   wins+=outcome=='win';draws+=outcome=='draw';losses+=outcome=='loss'
   rows.append({'position_id':position_id,'game':game,'candidate_first':candidate_seat==initial.turn,'candidate_seat_red':candidate_seat==1,'outcome':outcome,'candidate_nodes':candidate_nodes,'baseline_nodes':baseline_nodes})
 return {'opponent':'default handcrafted weights','heldout_positions':len(heldout),'matches_per_position':matches_per_position,'depth':depth,'wins':wins,'draws':draws,'losses':losses,'matches':len(rows),'results':rows,'limitation':'Small deterministic set of four held-out reachable states; outcomes measure this search setup and opponent only, not general playing strength.'}

def main():
 result=genetic_optimize();result['heldout_tournament']=heldout_tournament(result['weights'],matches_per_position=2,depth=1);root=MODEL_DIR;root.mkdir(parents=True,exist_ok=True)
 weight_path=root/'genetic_weights.json';weight_path.write_text(json.dumps(result['weights'],indent=2))
 result['artifact_metadata']={'weights_file':weight_path.name,'weights_sha256':hashlib.sha256(weight_path.read_bytes()).hexdigest(),'training_command':'python -m app.training.train_genetic'}
 (root/'genetic_experiment.json').write_text(json.dumps({k:v for k,v in result.items() if k!='weights'},indent=2))
 print(json.dumps({'weights':result['weights'],'config':result['config'],'fitness_history':result['fitness_history']},indent=2))
if __name__=='__main__':main()
