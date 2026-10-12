import json, random
from pathlib import Path
from ..games.connect_four import ConnectFour

DEFAULT_WEIGHTS={'center':3.0,'open1':0.2,'open2':1.1,'open3':8.0,'threats':18.0}

def features(state):
    b=state.board; f={k:0.0 for k in DEFAULT_WEIGHTS}
    center=[b[r][3] for r in range(6)];f['center']=sum(center)
    windows=[]
    for r in range(6):
      for c in range(7):
       for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
        coords=[(r+i*dr,c+i*dc) for i in range(4)]
        if all(0<=rr<6 and 0<=cc<7 for rr,cc in coords):windows.append([b[rr][cc] for rr,cc in coords])
    for w in windows:
      red=w.count(1);yellow=w.count(-1)
      if not yellow and red in (1,2,3):f[f'open{red}']+=1
      if not red and yellow in (1,2,3):f[f'open{yellow}']-=1
    # Immediate next-turn winning drops, including gravity-valid positions.
    for col in state.legal_moves():
      cp=state.clone();cp.turn=1;cp.play(col)
      if cp.winner()==1:f['threats']+=1
      cp=state.clone();cp.turn=-1;cp.play(col)
      if cp.winner()==-1:f['threats']-=1
    return f

def handcrafted(state,weights=None):
    w=weights or DEFAULT_WEIGHTS
    return sum(features(state)[k]*w[k] for k in DEFAULT_WEIGHTS)

def save_weights(weights,path): Path(path).write_text(json.dumps(weights,indent=2))
def load_weights(path): return json.loads(Path(path).read_text())

def genetic_optimize(generations=8,population_size=20,seed=7):
    """Evolve move-evaluation weights on fixed tactical positions and opponent profiles."""
    rng=random.Random(seed); keys=list(DEFAULT_WEIGHTS); positions=_fixed_positions()
    train_positions=positions[:8]; heldout_positions=positions[8:12]
    opponents=[
      {'center':3.0,'open1':0.2,'open2':1.1,'open3':8.0,'threats':18.0},
      {'center':1.0,'open1':0.1,'open2':2.0,'open3':12.0,'threats':30.0},
    ]
    heldout_opponents=[
      {'center':5.0,'open1':0.1,'open2':1.5,'open3':10.0,'threats':24.0},
    ]
    pop=[{k:rng.uniform(0.1,20) for k in keys} for _ in range(population_size)]
    def fitness(w, samples, opponent_profiles):
      total=0.0;cases=0
      for state in samples:
       moves=state.legal_moves()
       if not moves:continue
       candidate=[]
       for col in moves:
        child=state.clone();child.play(col)
        candidate.append((handcrafted(child,w)*state.turn,col,child))
       _,_,chosen=max(candidate,key=lambda item:(item[0],-item[1]))
       for opponent_weights in opponent_profiles:
        # Opponent-profile utility is scored in the candidate mover's perspective.
        total += handcrafted(chosen,opponent_weights)*state.turn
        cases += 1
      return total/max(1,cases)
    history=[]
    for _ in range(generations):
      ranked=sorted(((fitness(x,train_positions,opponents),x) for x in pop),key=lambda x:x[0],reverse=True)
      history.append(ranked[0][0]);nextpop=[ranked[0][1].copy(),ranked[1][1].copy()]
      while len(nextpop)<population_size:
       a=rng.choice(ranked[:max(2,population_size//2)])[1];b=rng.choice(ranked[:max(2,population_size//2)])[1]
       child={k:(a[k] if rng.random()<.5 else b[k]) for k in keys}
       if rng.random()<.25:
        k=rng.choice(keys);child[k]=max(0.01,child[k]+rng.gauss(0,2))
       nextpop.append(child)
      pop=nextpop
    best=max(pop,key=lambda x:fitness(x,train_positions,opponents))
    return {'weights':best,'config':{'generations':generations,'population_size':population_size,'seed':seed,'opponents':['default handcrafted tactical profile','threat-weighted handcrafted profile'],'fitness_definition':'one-ply proxy: mean successor heuristic score after the candidate move, using fixed opponent weight profiles and candidate-mover perspective; no opponent reply or game outcome is simulated','train_position_count':len(train_positions),'heldout_opponent_profiles':['heldout balanced profile'],'heldout_position_count':len(heldout_positions)},'fitness_history':history,'heldout_proxy_fitness':fitness(best,heldout_positions,heldout_opponents)}

def _fixed_positions():
    """Seeded, realistic reachable midgame positions used only to compare chromosomes."""
    rng=random.Random(73);out=[];seen=set()
    for _ in range(100):
      s=ConnectFour()
      for _ply in range(rng.randint(5,16)):
        if s.terminal():break
        s.play(rng.choice(s.legal_moves()))
      if s.terminal():continue
      key=min(s.key(),s.mirrored_key())
      if key not in seen:seen.add(key);out.append(s)
      if len(out)==12:break
    return out
