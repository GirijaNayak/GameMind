export type Game = 'connect4'|'tictactoe'|'nim';
export type Evaluator = 'handcrafted'|'genetic'|'neural'|'optimal';
export interface SearchResult {move:number|string|[number,number]; value:number; nodes:number; elapsed_ms:number; completed_depth:number; candidate_scores:Record<string,number>; pruned:number; evaluator:string; game:string}
export interface HistoryRecord {created_utc:string;records:number;pending:string[];results:unknown[]}
