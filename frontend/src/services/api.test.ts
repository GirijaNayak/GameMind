import {expect,it} from 'vitest';
import {search} from './api';

it('sends game state and AI settings to the backend',async()=>{
 let sent:Record<string,unknown>|undefined;
 const oldFetch=globalThis.fetch;
 globalThis.fetch=(async(_url:RequestInfo|URL,init?:RequestInit)=>{sent=JSON.parse(String(init?.body));return {ok:true,json:async()=>({move:3,nodes:7})} as Response}) as typeof fetch;
 try {
  await search('connect4',Array.from({length:6},()=>Array(7).fill(0)),1,4,'neural',50);
  expect(sent).toMatchObject({game:'connect4',turn:1,depth:4,evaluator:'neural',time_limit_ms:50});
  expect(sent?.board).toHaveLength(6);
 } finally {globalThis.fetch=oldFetch}
});
