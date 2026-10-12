import random
from ..games.connect_four import ConnectFour
from ..evaluators.neural import train_positions
from ..evaluators.heuristic import handcrafted
from ..paths import MODEL_DIR

def generate_training_data(games=100,seed=21):
 """Generate realized outcomes from completed, heuristic-guided self-play games."""
 rng=random.Random(seed);states=[];labels=[];groups=[]
 for game_id in range(games):
  s=ConnectFour();trajectory=[]
  while not s.terminal():
   trajectory.append(s.clone())
   moves=s.legal_moves();
   if rng.random()<.7:
    scores=[]
    for c in moves:
     t=s.clone();t.play(c);scores.append((handcrafted(t),c))
    c=max(scores)[1] if s.turn==1 else min(scores)[1]
   else:c=rng.choice(moves)
   s.play(c)
  winner=s.winner()
  for pos in trajectory:
   states.append(pos);labels.append(float(winner*pos.turn));groups.append(game_id)
 return states,labels,groups

def main():
 states,labels,groups=generate_training_data()
 dataset_metadata={'games_generated':100,'generation_seed':21,'source_group_ids':list(range(100)),'positions_removed_by_filter':0,'policy':'At each ply, choose a handcrafted-score greedy move with probability 0.7 and a random legal move with probability 0.3.'}
 print(train_positions(states,labels,output=MODEL_DIR/'connect4.pt',groups=groups,dataset_metadata=dataset_metadata))
if __name__=='__main__':main()

