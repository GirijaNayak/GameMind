import type {Evaluator,Game,SearchResult} from '../types';
const API=import.meta.env.VITE_API_URL??'http://localhost:8000';
export async function search(game:Game,board:number[][],turn:number,depth:number,evaluator:Evaluator,time_limit_ms?:number):Promise<SearchResult>{
 const body:Record<string,unknown>={game,turn,depth,evaluator}; if(game==='connect4')body.board=board;else if(game==='tictactoe')body.cells=board.flat();else body.piles=board[0];
 if(time_limit_ms)body.time_limit_ms=time_limit_ms;
 const res=await fetch(`${API}/api/search`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 if(!res.ok)throw new Error((await res.json()).detail??`API ${res.status}`);return res.json();
}
export async function runTournament(matches:number,depth:number,budget:'depth'|'nodes'|'time',budgetValue?:number,seed=20261011){const res=await fetch(`${API}/api/experiments/tournament`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({matches,depth,seed,budget,budget_value:budgetValue})});if(!res.ok)throw new Error('Experiment request failed');return res.json();}
export async function getHistory(){const r=await fetch(`${API}/api/experiments/history`);if(!r.ok)throw new Error('History unavailable');return r.json();}
