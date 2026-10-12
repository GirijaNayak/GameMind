"""Optional PyTorch evaluator. Encoding is 2x6x7 (current side, opponent); tanh value in [-1,1]."""
from pathlib import Path
import hashlib
import numpy as np
try:
 import torch
 from torch import nn
 class ValueNet(nn.Module):
  def __init__(self):
   super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(84,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1),nn.Tanh())
  def forward(self,x):return self.net(x)
except ImportError:
 torch=None;ValueNet=None

def encode(state):
 p=state.turn;return np.array([[[1 if x==p else 0 for x in row] for row in state.board],[[1 if x==-p else 0 for x in row] for row in state.board]],dtype=np.float32)

class NeuralEvaluator:
 def __init__(self,path=None):
  if torch is None:raise RuntimeError('Install backend requirements to enable PyTorch')
  self.model=ValueNet();self.path=Path(path) if path else Path(__file__).resolve().parents[2]/'models'/'connect4.pt'
  if not self.path.is_file():raise FileNotFoundError(f'Neural checkpoint not found: {self.path}')
  self.model.load_state_dict(torch.load(self.path,map_location='cpu',weights_only=True))
  self.model.eval()
 def __call__(self,state):
  with torch.no_grad():return float(self.model(torch.from_numpy(encode(state)[None])).item())*100

def train_positions(positions,labels,output=None,epochs=30,seed=11,groups=None,dataset_metadata=None):
 if torch is None:raise RuntimeError('PyTorch is required for training')
 if len(positions)!=len(labels):raise ValueError('positions and labels must have equal length')
 if groups is not None and len(groups)!=len(positions):raise ValueError('groups must match positions')
 dataset_metadata=dataset_metadata or {}
 source_groups=list(groups) if groups is not None else list(range(len(positions)))
 torch.manual_seed(seed);xs=[];ys=[];sample_groups=[];unique_states=[];seen=set()
 for state,label,group in zip(positions,labels,source_groups):
  encoded=encode(state);mirrored=encoded[:,:,::-1].copy();key=min(encoded.tobytes(),mirrored.tobytes())
  if key in seen:continue
  seen.add(key);xs.append(encoded);ys.append(float(label));sample_groups.append(group);unique_states.append(state)
 if len(xs)<3:raise ValueError('Need at least three unique positions')
 # Canonicalize exact and mirrored boards globally before assigning whole games to splits.
 unique_groups=list(dict.fromkeys(sample_groups));rng=np.random.default_rng(seed);rng.shuffle(unique_groups)
 if len(unique_groups)<3:raise ValueError('Need at least three independent groups for train/validation/test')
 n_groups=len(unique_groups);n_train_groups=max(1,int(.70*n_groups));n_val_groups=max(1,int(.15*n_groups))
 if n_train_groups+n_val_groups>=n_groups:n_train_groups=n_groups-2;n_val_groups=1
 train_groups=set(unique_groups[:n_train_groups]);val_groups=set(unique_groups[n_train_groups:n_train_groups+n_val_groups]);test_groups=set(unique_groups[n_train_groups+n_val_groups:])
 split_groups={'train':train_groups,'validation':val_groups,'test':test_groups}
 split_indices={name:np.array([i for i,g in enumerate(sample_groups) if g in group_set]) for name,group_set in split_groups.items()}
 tr=split_indices['train'];va=split_indices['validation'];te=split_indices['test']
 x=torch.tensor(np.stack(xs));y=torch.tensor(ys,dtype=torch.float32).reshape(-1,1);model=ValueNet();opt=torch.optim.Adam(model.parameters(),lr=0.001);loss_fn=torch.nn.MSELoss()
 for _ in range(epochs):
  model.train();opt.zero_grad();loss=loss_fn(model(x[tr]),y[tr]);loss.backward();opt.step()
 model.eval();train_mse=float(loss_fn(model(x[tr]),y[tr]).item());val=float(loss_fn(model(x[va]),y[va]).item());test=float(loss_fn(model(x[te]),y[te]).item())
 train_mean=float(y[tr].mean().item())
 baseline_val=float(loss_fn(torch.full_like(y[va],train_mean),y[va]).item());baseline_test=float(loss_fn(torch.full_like(y[te],train_mean),y[te]).item())
 from .heuristic import handcrafted
 handcrafted_predictions=np.array([np.sign(handcrafted(state)*state.turn) for state in unique_states],dtype=np.float32)
 target_array=np.asarray(ys,dtype=np.float32)
 handcrafted_mse={name:float(np.mean((handcrafted_predictions[index]-target_array[index])**2)) for name,index in split_indices.items()}
 target_distribution={name:{'negative':int(np.sum(target_array[index]<0)),'zero':int(np.sum(target_array[index]==0)),'positive':int(np.sum(target_array[index]>0)),'mean':float(np.mean(target_array[index])),'std':float(np.std(target_array[index]))} for name,index in split_indices.items()}
 with torch.no_grad():all_predictions=model(x).reshape(-1).cpu().numpy()
 expected_groups=dataset_metadata.get('source_group_ids',list(dict.fromkeys(source_groups)))
 missing_groups=[group for group in expected_groups if group not in set(sample_groups)]
 dataset_counts={'games_generated':dataset_metadata.get('games_generated',len(set(source_groups))),'generation_seed':dataset_metadata.get('generation_seed'),'games_with_raw_positions':len(set(source_groups)),'positions_before_deduplication':len(positions),'positions_removed_by_filter':dataset_metadata.get('positions_removed_by_filter',0),'positions_removed_by_exact_or_mirror_deduplication':len(positions)-len(xs),'positions_after_deduplication':len(xs),'games_with_retained_positions':len(unique_groups),'games_with_no_retained_positions':missing_groups,'games_by_split':{name:len(group_set) for name,group_set in split_groups.items()},'positions_by_split':{name:len(index) for name,index in split_indices.items()}}
 p=Path(output) if output else Path(__file__).resolve().parents[2]/'models'/'connect4.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),p)
 meta={'architecture':'84-128-64-1 MLP ReLU/Tanh','encoding':'two 6x7 occupancy planes: side-to-move and opponent; same encode() is used for training and inference','labels':'realized terminal outcomes of completed heuristic-guided self-play trajectories; these are not exact game-theoretic values','target_scale':'winner * side_to_move, yielding -1 for a realized loss and +1 for a realized win (0 for draws)','inference_scale':'ValueNet tanh output is in [-1,1]; NeuralEvaluator scales by 100 and the API/tournament multiply by state.turn to convert side-to-move output to Red-perspective search value','observed_model_output_range':[float(all_predictions.min()),float(all_predictions.max())],'loss':'mean squared error on side-to-move targets','optimizer':'Adam lr=0.001','training_epochs':epochs,'checkpoint_selection':'final fixed epoch; validation and test metrics do not select or early-stop training','training_split_seed':seed,'dataset_generation':dataset_metadata.get('policy'),'dataset_counts':dataset_counts,'target_distribution_by_split':target_distribution,'unique_positions':len(xs),'train_positions':len(tr),'validation_positions':len(va),'test_positions':len(te),'unique_train_groups':len(train_groups),'unique_validation_groups':len(val_groups),'unique_test_groups':len(test_groups),'training_mse':train_mse,'validation_mse':val,'test_mse':test,'training_mean_baseline_validation_mse':baseline_val,'training_mean_baseline_test_mse':baseline_test,'handcrafted_sign_baseline_mse_by_split':handcrafted_mse,'split':'70/15/15 by source game after global mirror canonicalization and deduplication','duplicate_policy':'deduplicate exact and horizontally mirrored encodings globally before splitting','checkpoint_file':p.name,'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 p.with_suffix('.json').write_text(__import__('json').dumps(meta,indent=2));return meta
