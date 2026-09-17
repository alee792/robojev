"""Live view: judgments with probabilities and confidence, latency, entities, camera frame, and
inputs for standing orders, task text, STOP and RESUME. One page, polled at 4 Hz."""
from __future__ import annotations

import json

from aiohttp import web

PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>robojev</title>
<style>
:root{
 --bg:#0e0f11;--panel:#16181b;--line:#282c31;--sep:#1e2126;--fg:#e7e9ec;--muted:#8a929b;
 --ok:#5fd08a;--warn:#e0a850;--bad:#ff6058;--pend:#d8d05a;--accent:#7f9ad6;
 --lg:26px;--md:15px;--sm:12px;
 --font:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--font);font-size:var(--md);line-height:1.45;
 font-variant-numeric:tabular-nums}
h2{margin:0 0 8px;font-size:var(--sm);font-weight:600;color:var(--muted);letter-spacing:.04em}
h2.sec{margin-top:14px}
.muted{color:var(--muted)}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.pend{color:var(--pend)}
.ch{color:#fff;font-weight:600}
.hidden{display:none}

/* top band ------------------------------------------------------------------ */
#band{background:var(--panel);border-bottom:1px solid var(--line);padding:12px 14px;
 display:flex;flex-wrap:wrap;gap:10px 30px;align-items:flex-start}
.stat{display:flex;flex-direction:column;min-width:0}
.stat .lbl{font-size:var(--sm);color:var(--muted)}
.stat .val{font-size:var(--lg);line-height:1.15;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat .sub{font-size:var(--sm);color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat.wide{flex:1 1 280px}
.stat.wide .val{white-space:normal}
#chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center;align-self:center}
.chip{font-size:var(--sm);padding:2px 9px;border-radius:999px;border:1px solid currentColor}
#ctl{margin-left:auto;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
#why{flex-basis:100%;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline;
 font-size:var(--sm);border-top:1px solid var(--line);padding-top:8px}

/* panels -------------------------------------------------------------------- */
main{display:grid;grid-template-columns:minmax(300px,1fr) minmax(400px,1.3fr) minmax(300px,1fr);
 gap:12px;padding:12px;align-items:start}
#bottom{padding:0 12px 12px}
@media(max-width:1180px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.card+.card{margin-top:12px}

/* controls ------------------------------------------------------------------ */
button{font-family:inherit;font-size:var(--md);padding:8px 14px;background:#22262b;color:var(--fg);
 border:1px solid #3a4048;border-radius:6px;cursor:pointer}
button:hover{background:#2b3037}
button.stop{background:#5c1a17;border-color:var(--bad);color:#ffdedb;font-weight:700}
button.go{background:#173d28;border-color:var(--ok)}
input,textarea{width:100%;font-family:inherit;font-size:var(--sm);background:#0b0c0e;color:var(--fg);
 border:1px solid #3a4048;border-radius:6px;padding:6px 8px}
.row{margin-top:8px}

/* tables -------------------------------------------------------------------- */
table{border-collapse:collapse;width:100%;font-size:var(--sm)}
th{text-align:left;font-weight:600;color:var(--muted);padding:4px 6px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:4px 6px;vertical-align:top;border-bottom:1px solid var(--sep)}
.dist{display:flex;flex-direction:column;gap:2px}
.d{display:flex;align-items:center;gap:6px;white-space:nowrap}
.bar{flex:0 0 auto;width:54px;height:5px;background:#2a2e33;border-radius:3px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font-size:var(--sm)}
.kv .k{color:var(--muted)}

/* media and blobs ----------------------------------------------------------- */
img.cam{display:block;width:100%;border-radius:6px;background:#000}
canvas{display:block;width:100%;height:40px;background:#0b0c0e;border-radius:4px;margin-top:8px}
pre{margin:0;font-family:inherit;font-size:var(--sm);color:var(--muted);white-space:pre-wrap;
 word-break:break-word;max-height:220px;overflow:auto}
#per{max-height:120px}
details{margin-top:12px;border-top:1px solid var(--line);padding-top:8px}
summary{cursor:pointer;font-size:var(--sm);color:var(--muted)}
#graph{width:100%;height:340px;background:var(--bg);border:1px solid var(--line);border-radius:6px}
</style></head><body>

<header id=band>
<div class=stat><span class=lbl>Arm</span><span class=val id=v_arm>&mdash;</span><span class=sub id=v_arm_sub></span></div>
<div class=stat><span class=lbl>Ladder</span><span class=val id=v_ladder>&mdash;</span><span class=sub id=v_ladder_sub></span></div>
<div class=stat><span class=lbl>Primitive</span><span class=val id=v_prim>&mdash;</span><span class=sub id=v_prim_sub></span></div>
<div class=stat><span class=lbl>Target</span><span class=val id=v_target>&mdash;</span></div>
<div class=stat><span class=lbl>Place</span><span class=val id=v_place>&mdash;</span></div>
<div class="stat wide"><span class=lbl>Task</span><span class=val id=v_task>&mdash;</span></div>
<div id=chips>
 <span class="chip bad hidden" id=c_error></span>
 <span class="chip warn hidden" id=c_paused>Jev paused</span>
 <span class="chip bad hidden" id=c_override>Override</span>
 <span class="chip bad hidden" id=c_evade></span>
 <span class="chip ok hidden" id=c_done>Task done</span>
</div>
<div id=ctl>
 <button class=stop onclick="post('/api/stop')">Stop (freeze)</button>
 <button class=go onclick="post('/api/resume')">Resume</button>
 <button onclick="post('/api/pause')">Pause Jev</button>
 <button onclick="post('/api/unpause')">Unpause</button>
</div>
<div id=why><span id=v_reason class=muted></span><span id=v_age class=muted></span></div>
</header>

<main>
<section>
 <div class=card>
  <h2>Wrist camera</h2>
  <img id=cam class="cam hidden" src="/frame.jpg"><div class="noframe muted" id=cam_no>No frame</div>
  <h2 class=sec>Overhead camera</h2>
  <img id=over class="cam hidden" src="/frame2.jpg"><div class="noframe muted" id=over_no>No frame</div>
  <h2 class=sec>Third camera</h2>
  <img id=third class="cam hidden" src="/third.jpg"><div class="noframe muted" id=third_no>No frame</div>
 </div>
 <div class=card><h2>Arm detail</h2><div class=kv id=armkv></div></div>
 <div class=card>
  <h2>User task (separate from orders)</h2>
  <input id=task placeholder="hover over the paper cup">
  <div class=row><button onclick="post('/api/task',{text:g('task').value})">Set task</button></div>
  <h2 class=sec>Standing orders (operator, one per line)</h2>
  <textarea id=orders rows=4></textarea>
  <div class=row><button onclick="post('/api/orders',{orders:g('orders').value.split('\n')})">Apply orders</button></div>
 </div>
</section>

<section>
 <div class=card><h2>Judgments <span id=qs></span></h2><table id=j></table></div>
</section>

<section>
 <div class=card><h2>Brain</h2><div id=brain></div></div>
 <div class=card><h2>Jev</h2><div id=stats class=muted></div><canvas id=spark width=300 height=40></canvas></div>
 <div class=card><h2>Events</h2><pre id=ev></pre></div>
 <div class=card><h2>Objects (base frame, metres)</h2><table id=ent></table></div>
</section>
</main>

<section id=bottom><div class=card>
 <h2>Graph</h2>
 <svg id=graph viewBox="0 0 1200 400" preserveAspectRatio="xMidYMid meet">
 <defs><marker id=arrow markerWidth=8 markerHeight=8 refX=7 refY=4 orient=auto><path d="M0,0 L8,4 L0,8 z" fill="#78859c"/></marker>
 <marker id=arrowM markerWidth=8 markerHeight=8 refX=7 refY=4 orient=auto><path d="M0,0 L8,4 L0,8 z" fill="#f0f"/></marker></defs>
 <g id=graphc></g></svg>
 <details>
  <summary>Perception and state sent to Jev</summary>
  <h2 class=sec>Perception</h2><pre id=per></pre>
  <h2 class=sec>State sent to Jev</h2><pre id=state></pre>
 </details>
</div></section>

<script>
const g=id=>document.getElementById(id);
async function post(u,b){await fetch(u,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b||{})})}
let seeded=false;
function bar(p){return `<span class=bar><i style="width:${Math.round(p*100)}%"></i></span>`}

// ---- small formatters: every field may be null (no Jev, run stopped), so never throw. ----
const txt=v=>(v==null||v==='')?'—':String(v);
const num=(v,d,suf)=>(typeof v=='number'&&isFinite(v))?v.toFixed(d)+(suf||''):'—';
const vec=(v,d)=>Array.isArray(v)?v.map(x=>typeof x=='number'?x.toFixed(d):txt(x)).join('  '):'—';
const cm=v=>typeof v=='number'?v*100:null;
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function val(id,text,cls){const e=g(id);e.textContent=text;e.className='val'+(cls?' '+cls:'')}
function sub(id,text){g(id).textContent=text||''}
function chip(id,on,text){const e=g(id);if(text!=null)e.textContent=text;e.classList.toggle('hidden',!on)}
function statusClass(a){return a=='applied'?'ok':a=='override'?'bad':a=='gated'?'warn':a=='pending_confirmation'?'pend':'muted'}

// ---- GRAPH panel: rebuilds the Doom-style composition graph from the snapshot every poll. ----
function gesc(v){return String(v==null?'—':v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function gtrunc(v,n){const s=String(v==null?'—':v);return s.length>n?s.slice(0,n-1)+'…':s}
function judgeColor(applied){
  if(applied=='applied')return{fill:'#12281c',stroke:'#4a8'};
  if(applied=='override')return{fill:'#2a1414',stroke:'#f55'};
  if(applied=='gated')return{fill:'#2a1f10',stroke:'#c84'};
  if(applied=='pending_confirmation')return{fill:'#2a2810',stroke:'#cc4'};
  if(applied=='busy')return{fill:'#222',stroke:'#888'};
  return{fill:'#1b1b1b',stroke:'#669'};
}
function layoutGraph(nodes,edges){
  const H=400,colX=[30,250,500,760,990],colW=[190,220,230,200,180],boxH=40;
  const byCol=[[],[],[],[],[]];
  for(const id in nodes)byCol[nodes[id].col].push(nodes[id]);
  byCol.forEach((arr,col)=>{const n=arr.length;if(!n)return;const gap=(H-40)/(n+1);
    arr.forEach((nd,i)=>{nd.x=colX[col];nd.w=colW[col];nd.cy=20+gap*(i+1);nd.y=nd.cy-boxH/2;nd.h=boxH})});
  let svg='';
  for(const e of edges){
    const f=nodes[e.f],t=nodes[e.t]; if(!f||!t)continue;
    const x1=f.x+f.w,y1=f.cy,x2=t.x,y2=t.cy,mx=(x1+x2)/2;
    const stroke=e.override?'#f0f':'#78859c';
    svg+=`<path d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" fill=none stroke="${stroke}" stroke-width="${e.override?2:1.2}" ${e.override?'stroke-dasharray="6,4"':''} opacity="${e.override?1:0.75}" marker-end="url(#${e.override?'arrowM':'arrow'})"/>`;
    if(e.label)svg+=`<text x="${mx}" y="${(y1+y2)/2-5}" fill="#f0f" font-size="9" text-anchor="middle">${gesc(e.label)}</text>`;
  }
  for(const id in nodes){
    const nd=nodes[id];
    const fill=nd.fill||(nd.green?'#12281c':'#1b1b1b');
    const stroke=nd.stroke||(nd.green?'#4a8':(nd.dashed?'#78859c':'#556'));
    const dashattr=nd.dashed?'stroke-dasharray="4,3"':'';
    if(nd.shape=='oval')svg+=`<ellipse cx="${nd.x+nd.w/2}" cy="${nd.cy}" rx="${nd.w/2}" ry="${nd.h/2}" fill="${fill}" stroke="${stroke}" ${dashattr}/>`;
    else svg+=`<rect x="${nd.x}" y="${nd.y}" width="${nd.w}" height="${nd.h}" rx="6" fill="${fill}" stroke="${stroke}" ${dashattr}/>`;
    svg+=`<text x="${nd.x+nd.w/2}" y="${nd.cy-(nd.sub?4:-3)}" fill="#eee" font-size="10" text-anchor="middle" font-weight="600">${gesc(nd.label)}</text>`;
    if(nd.sub)svg+=`<text x="${nd.x+nd.w/2}" y="${nd.cy+10}" fill="#9aa" font-size="9" text-anchor="middle">${gesc(nd.sub)}</text>`;
  }
  return svg;
}
function buildGraph(s){
  try{
    const b=s.brain||{},j=s.judgments||{},a=s.arm||{};
    const nodes={},edges=[];
    function node(id,col,label,opts){nodes[id]=Object.assign({id,col,label},opts||{})}
    function edge(f,t,opts){edges.push(Object.assign({f,t},opts||{}))}
    // column 1: state inputs (dashed)
    node('in_obj',0,`objects: ${(s.entities||[]).length}`,{shape:'rect',dashed:true});
    node('in_orders',0,`standing orders: ${(s.orders||[]).length}`,{shape:'rect',dashed:true});
    node('in_task',0,`user request`,{shape:'rect',dashed:true,sub:gtrunc(s.task||'—',26)});
    // column 2: code-built option sets (dashed)
    if(j.target)node('opt_target',1,`target options: ${Object.keys(j.target.probabilities||{}).length}`,{shape:'rect',dashed:true});
    if(j.place)node('opt_place',1,`place options: ${Object.keys(j.place.probabilities||{}).length}`,{shape:'rect',dashed:true});
    node('opt_prim',1,`offered primitives: ${(s.offered||[]).length}`,{shape:'rect',dashed:true});
    // column 3: Jev answers (orange, coloured by applied status)
    for(const k of ['target','place','next','motion','hover_position','hover_height','speed','avoid','orders_violated','task_done']){
      const jj=j[k]; if(!jj)continue;
      const c=judgeColor(jj.applied);
      node('q_'+k,2,`${k}: ${gtrunc(jj.chosen,16)}`,{shape:'rect',fill:c.fill,stroke:c.stroke,
        sub:`p ${jj.p!=null?jj.p.toFixed(2):'—'}${jj.confidence!=null?' c '+jj.confidence.toFixed(2):''} · ${jj.applied}`});
    }
    // column 4: code transforms (ovals)
    node('tr_target',3,'target → subject',{shape:'oval',sub:gtrunc(b.target,18)});
    node('tr_place',3,'place → xy',{shape:'oval',sub:gtrunc(b.place,18)});
    node('tr_prim',3,'primitive → goal',{shape:'oval',sub:gtrunc(`${b.prim||'—'} ${b.prim_subject||''}`.trim(),20)});
    if(b.ladder&&b.ladder!='fresh')node('ladder',3,'silence ladder',{shape:'oval',stroke:'#f55',sub:b.ladder});
    // column 5: actuator command (green)
    const goal=(a.goal||[]).map(v=>typeof v=='number'?v.toFixed(2):v).join(', ');
    node('cmd_move',4,'MOVE',{shape:'rect',green:true,sub:`${goal||'—'} @ ${a.speed_cap!=null?(a.speed_cap*100).toFixed(0)+' cm/s':'—'}`});
    node('cmd_grip',4,'GRIPPER',{shape:'rect',green:true,sub:a.gripper!=null?a.gripper.toFixed(3):'—'});
    // edges: inputs -> option sets -> answers -> transforms -> command
    edge('in_obj','opt_target');edge('in_obj','opt_place');edge('in_task','opt_target');edge('in_task','opt_prim');edge('in_orders','opt_prim');
    edge('opt_target','q_target');edge('opt_place','q_place');edge('opt_prim','q_next');edge('in_orders','q_orders_violated');
    edge('q_target','tr_target');edge('q_place','tr_place');edge('q_next','tr_prim');
    edge('q_motion','tr_prim');edge('q_hover_position','tr_prim');edge('q_hover_height','tr_prim');
    edge('q_speed','cmd_move');
    edge('tr_target','cmd_move');edge('tr_place','cmd_move');edge('tr_prim','cmd_move');edge('tr_prim','cmd_grip');
    // overrides: avoid / orders_violated / silence ladder win over the composed command (magenta dashed)
    if(b.avoid)edge('q_avoid','cmd_move',{override:true,label:'avoid '+b.avoid});
    if(b.override)edge('q_orders_violated','cmd_move',{override:true,label:'orders_violated'});
    if(b.ladder&&b.ladder!='fresh')edge('ladder','cmd_move',{override:true,label:'ladder: '+b.ladder});
    return layoutGraph(nodes,edges);
  }catch(e){return `<text x=10 y=20 fill="#f66" font-size="11">graph error: ${gesc(e.message)}</text>`}
}
function render(s){
const a=s.arm||{},b=s.brain||{},t=s.stats||{},js=s.judgments||{};
// ---- top band: what a visitor reads from a metre away ----
const frozen=!!a.frozen;
val('v_arm',frozen?'frozen':txt(a.status),frozen?'bad':(a.status=='live'?'ok':'warn'));
sub('v_arm_sub','cap '+num(cm(a.speed_cap),0,' cm/s'));
val('v_ladder',txt(b.ladder),b.ladder=='fresh'?'ok':'warn');
sub('v_ladder_sub','answer age '+num(b.answer_age_s,2,' s'));
val('v_prim',[b.prim,b.prim_subject].filter(Boolean).join(' ')||'—');
sub('v_prim_sub',[txt(b.prim_status),b.prim_age!=null?num(b.prim_age,1,' s'):''].filter(x=>x&&x!='—').join(' · '));
val('v_target',txt(b.target));
val('v_place',txt(b.place));
val('v_task',txt(s.task));
chip('c_error',!!a.error,a.error||'');
chip('c_paused',!!s.paused);
chip('c_override',!!b.override);
chip('c_evade',!!(b.avoid||b.evade),'Evade '+(b.evade||('away from '+b.avoid)));
chip('c_done',!!b.done);
g('v_reason').textContent=txt(s.reason);
// ---- arm detail ----
const rows=[['ee',vec(a.ee,3)],['setpoint',vec(a.setpoint,3)],
  ['goal',vec(a.goal,3)+'  @ '+num(cm(a.speed_cap),0,' cm/s')],
  ['F_ext',vec(a.ext_force,1)+' N'],['gripper',num(a.gripper,3)]];
if(a.pitch!=null)rows.push(['wrist pitch',num(a.pitch,2,' rad')]);
g('armkv').innerHTML=rows.map(([k,v])=>`<span class=k>${esc(k)}</span><span>${esc(v)}</span>`).join('');
// ---- judgments ----
g('qs').textContent=`${txt(s.question_set)} · ${txt(s.model)} · in flight ${s.in_flight==null?'—':s.in_flight}${s.paused?' · paused':''}`;
let h='<tr><th>Question</th><th>Pick</th><th>p / conf</th><th>Status</th><th>Age</th><th>Distribution</th></tr>';
for(const k in js){const q=js[k];
  const ps=Object.entries(q.probabilities||{}).sort((x,y)=>y[1]-x[1]).slice(0,5);
  const d=ps.map(([o,p])=>`<div class=d>${bar(p)}<span class="${o==q.chosen?'ch':''}">${esc(o)}</span><span class=muted>${num(p,2)}</span></div>`).join('');
  h+=`<tr><td>${esc(k)}</td><td class=ch>${esc(txt(q.chosen))}</td>`+
     `<td>${num(q.p,2)}${q.confidence!=null?' / '+num(q.confidence,2):''}</td>`+
     `<td class=${statusClass(q.applied)}>${esc(txt(q.applied))}</td>`+
     `<td>${num(q.age_ms,0,' ms')}</td><td><div class=dist>${d}</div></td></tr>`}
g('j').innerHTML=h;
g('graphc').innerHTML=buildGraph(s);
// ---- brain, jev, events, objects ----
g('brain').innerHTML=`speed <b>${esc(txt(b.speed_name))}</b> · motion <b>${esc(txt(b.motion))}</b> · holding <b>${esc(txt(b.held))}</b><br>`+
 `last result: ${esc(txt(b.last_result))}<br>offered: ${esc((s.offered||[]).join(', ')||'—')}<br>`+
 `recent: ${esc((b.recent||[]).join(' → ')||'—')}`;
g('stats').innerHTML=`sent ${txt(t.sent)} · ok ${txt(t.ok)} · errors ${txt(t.errors)} · timeouts ${txt(t.timeouts)} · `+
 `skipped ${txt(t.skipped)} · stale ${txt(t.dropped_stale)} · out of order ${txt(t.dropped_order)}<br>`+
 `latency p50 ${num(t.latency_p50,0)} p95 ${num(t.latency_p95,0)} max ${num(t.latency_max,0)} ms · `+
 `answer age p50 ${num(t.age_p50,0)} p95 ${num(t.age_p95,0)} ms · ${num(t.tokens/1e6,2)} Mtok`;
const c=g('spark').getContext('2d');c.clearRect(0,0,300,40);c.fillStyle='#7f9ad6';
(t.sparkline||[]).forEach((v,i)=>{const hh=Math.min(40,v/25);c.fillRect(i*5,40-hh,4,hh)});
let e='<tr><th>Label</th><th>Looks like</th><th>x</th><th>y</th><th>Horiz</th><th>Bearing</th><th>Status</th></tr>';
for(const o of (s.entities||[])){const xyz=o.xyz||[];
 e+=`<tr><td class=ch>${esc(txt(o.label))}</td><td>${esc(txt(o.description))}</td><td>${num(xyz[0],3)}</td><td>${num(xyz[1],3)}</td>`+
    `<td>${num(cm(o.horizontal_m),0,' cm')}</td><td>${num(o.bearing_deg,0,'°')}</td>`+
    `<td>${o.in_view?'in view':'out of view '+num(o.last_seen_s,0,' s')}${o.reachable?'':' <span class=bad>unreachable</span>'}</td></tr>`}
g('ent').innerHTML=e;
g('per').textContent=JSON.stringify(s.perception);
g('ev').textContent=(s.events||[]).slice().reverse().map(x=>new Date(x.t*1000).toLocaleTimeString()+' '+x.kind+' '+
 JSON.stringify(Object.fromEntries(Object.entries(x).filter(([k])=>k!='t'&&k!='kind')))).join('\n');
g('state').textContent=JSON.stringify(s.state,null,1);
if(!seeded){g('task').value=s.task||'';g('orders').value=(s.orders||[]).join('\n');seeded=true}
}
let lastOk=null;
async function poll(){try{const r=await fetch('/api/snapshot');const s=await r.json();lastOk=Date.now();render(s)}catch(e){console.log(e)}}
function freshness(){const e=g('v_age');
 if(lastOk==null){e.textContent='waiting for the run';e.className='warn';return}
 const dt=(Date.now()-lastOk)/1000;e.textContent='updated '+dt.toFixed(1)+' s ago';e.className=dt>2?'bad':'muted'}
// a camera endpoint with no frame answers with no image; hide the img instead of showing it broken.
for(const id of ['cam','over','third']){const im=g(id),no=g(id+'_no');
 im.addEventListener('load',()=>{im.classList.remove('hidden');no.classList.add('hidden')});
 im.addEventListener('error',()=>{im.classList.add('hidden');no.classList.remove('hidden')})}
setInterval(poll,250);setInterval(freshness,250);
setInterval(()=>{const t=Date.now();g('cam').src='/frame.jpg?'+t;g('third').src='/third.jpg?'+t;g('over').src='/frame2.jpg?'+t},200);
poll();freshness();
</script></body></html>"""


def make_app(loop) -> web.Application:
    app = web.Application()

    async def index(req):
        return web.Response(text=PAGE, content_type="text/html")

    async def snapshot(req):
        return web.Response(text=json.dumps(loop.view(), default=str), content_type="application/json")

    async def frame(req):
        jpg = loop.per.frame_jpeg
        if not jpg:
            return web.Response(status=204)
        return web.Response(body=jpg, content_type="image/jpeg")

    async def frame2(req):
        cams = loop.per.cameras
        jpg = loop.per.frames.get(cams[1][0]) if len(cams) > 1 else None
        if not jpg:
            return web.Response(status=204)
        return web.Response(body=jpg, content_type="image/jpeg")

    async def third(req):
        jpg = getattr(loop.per.cam, "third_jpeg", None)
        if not jpg:
            return web.Response(status=204)
        return web.Response(body=jpg, content_type="image/jpeg")

    async def orders(req):
        loop.set_orders((await req.json()).get("orders", [])); return web.json_response({"ok": True})

    async def task(req):
        loop.set_task((await req.json()).get("text", "")); return web.json_response({"ok": True})

    async def stop(req):
        loop.stop(); return web.json_response({"ok": True})

    async def resume(req):
        loop.resume(); return web.json_response({"ok": True})

    async def pause(req):
        loop.paused = True; loop.event("pause_jev"); return web.json_response({"ok": True})

    async def unpause(req):
        loop.paused = False; loop.event("unpause_jev"); return web.json_response({"ok": True})

    async def name(req):
        b = await req.json(); loop.per.tracker.set_name(b["id"], b.get("name")); loop.event("name", **b)
        return web.json_response({"ok": True})

    app.add_routes([web.get("/", index), web.get("/api/snapshot", snapshot), web.get("/frame.jpg", frame), web.get("/third.jpg", third), web.get("/frame2.jpg", frame2),
                    web.post("/api/orders", orders), web.post("/api/task", task), web.post("/api/stop", stop),
                    web.post("/api/resume", resume), web.post("/api/pause", pause), web.post("/api/unpause", unpause),
                    web.post("/api/name", name)])
    return app
