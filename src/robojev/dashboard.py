"""Live view: judgments with probabilities and confidence, latency, entities, camera frame, and
inputs for standing orders, task text, STOP and RESUME. One page, polled at 4 Hz."""
from __future__ import annotations

import json

from aiohttp import web

PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>robojev</title>
<style>
body{font:13px/1.35 -apple-system,Helvetica,Arial;margin:0;background:#111;color:#ddd}
.grid{display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:10px;padding:10px}
.card{background:#1b1b1b;border:1px solid #333;border-radius:6px;padding:8px}
h3{margin:0 0 6px;font-size:12px;color:#9ad;text-transform:uppercase;letter-spacing:.06em}
table{border-collapse:collapse;width:100%}td,th{padding:2px 4px;text-align:left;vertical-align:top}th{color:#888;font-weight:normal}
.bar{height:8px;background:#333;border-radius:3px;overflow:hidden;display:inline-block;width:110px;vertical-align:middle;margin-left:4px}
.bar i{display:block;height:100%;background:#4a8}
.ch{color:#fff;font-weight:600}.gated{color:#c84}.pending{color:#cc4}.override{color:#f55;font-weight:700}
.big{font-size:22px;color:#fff}.warn{color:#f66}.ok{color:#6c6}
textarea,input{width:100%;box-sizing:border-box;background:#000;color:#eee;border:1px solid #444;padding:4px;font:13px monospace}
button{padding:6px 12px;margin-right:6px;background:#2a2a2a;color:#eee;border:1px solid #555;border-radius:4px;cursor:pointer}
button.stop{background:#a22;border-color:#f44;font-weight:700}button.go{background:#262;border-color:#4a4}
pre{white-space:pre-wrap;font-size:11px;color:#aaa;max-height:260px;overflow:auto;margin:0}
img{width:100%;border-radius:4px}
canvas{background:#000;width:100%;height:40px}
.mono{font-family:monospace}
</style></head><body><div class=grid>
<div class=card><h3>arm</h3><div id=arm></div><div style="margin-top:8px">
<button class=stop onclick="post('/api/stop')">STOP (freeze)</button><button class=go onclick="post('/api/resume')">resume</button>
<button onclick="post('/api/pause')">pause Jev</button><button onclick="post('/api/unpause')">unpause</button></div>
<h3 style="margin-top:10px">user task (separate from orders)</h3><input id=task placeholder="hover over the paper cup"><button onclick="post('/api/task',{text:g('task').value})">set task</button>
<h3 style="margin-top:10px">standing orders (operator, one per line)</h3><textarea id=orders rows=4></textarea><button onclick="post('/api/orders',{orders:g('orders').value.split('\n')})">apply orders</button>
<h3 style="margin-top:10px">wrist camera</h3><img id=cam src="/frame.jpg"><img id=third src="/third.jpg" style="margin-top:6px"></div>
<div class=card><h3>judgments <span id=qs></span></h3><table id=j></table>
<h3 style="margin-top:10px">brain</h3><div id=brain class=mono></div>
<h3 style="margin-top:10px">jev</h3><div id=stats></div><canvas id=spark width=300 height=40></canvas>
<h3 style="margin-top:10px">events</h3><pre id=ev></pre></div>
<div class=card><h3>objects (base frame, metres)</h3><table id=ent></table>
<h3 style="margin-top:10px">perception</h3><div id=per class=mono></div>
<h3 style="margin-top:10px">state sent to jev</h3><pre id=state></pre></div>
</div><script>
const g=id=>document.getElementById(id);
async function post(u,b){await fetch(u,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b||{})})}
let seeded=false;
function bar(p){return `<span class=bar><i style="width:${Math.round(p*100)}%"></i></span>`}
async function poll(){try{const r=await fetch('/api/snapshot');const s=await r.json();
const a=s.arm,b=s.brain;
g('arm').innerHTML=`<span class=big ${a.frozen?'style="color:#f55"':''}>${a.status}${a.frozen?' FROZEN':''}</span> &nbsp; ladder: <b class=${b.ladder=='fresh'?'ok':'warn'}>${b.ladder}</b> &nbsp; answer age: ${b.answer_age_s==null?'—':b.answer_age_s.toFixed(2)+' s'}<br>
ee ${a.ee} &nbsp; setpoint ${a.setpoint}<br>goal ${a.goal} @ ${(a.speed_cap*100).toFixed(0)} cm/s<br>F_ext ${a.ext_force} N &nbsp; gripper ${a.gripper.toFixed(3)}<br><i>${s.reason}</i>${a.error?'<br><span class=warn>'+a.error+'</span>':''}`;
g('qs').textContent=`(${s.question_set}, ${s.model}, in flight ${s.in_flight}${s.paused?', PAUSED':''})`;
let h='<tr><th>question</th><th>pick</th><th>p / conf</th><th>distribution</th><th>status</th><th>age</th></tr>';
for(const k in s.judgments){const j=s.judgments[k];let d='';const ps=Object.entries(j.probabilities).sort((x,y)=>y[1]-x[1]).slice(0,5);
for(const [o,p] of ps)d+=`<div>${bar(p)} <span class=${o==j.chosen?'ch':''}>${o}</span> ${p.toFixed(2)}</div>`;
h+=`<tr><td>${k}</td><td class=ch>${j.chosen}</td><td>${j.p.toFixed(2)}${j.confidence!=null?' / '+j.confidence.toFixed(2):''}</td><td>${d}</td><td class=${j.applied=='applied'?'ok':j.applied=='override'?'override':j.applied=='gated'?'gated':'pending'}>${j.applied}</td><td>${Math.round(j.age_ms)} ms</td></tr>`}
g('j').innerHTML=h;
g('brain').innerHTML=`target <b>${b.target}</b> · motion <b>${b.motion}</b> · hover <b>${b.hover_position}</b>/<b>${b.hover_height}</b> · speed <b>${b.speed_name}</b>${b.override?' · <span class=override>OVERRIDE</span>':''}<br>recent: ${b.recent.join(' → ')}`;
const t=s.stats;g('stats').innerHTML=`sent ${t.sent} ok ${t.ok} err ${t.errors} timeouts ${t.timeouts} skipped ${t.skipped} stale ${t.dropped_stale} out-of-order ${t.dropped_order}<br>latency p50 ${t.latency_p50?.toFixed(0)} p95 ${t.latency_p95?.toFixed(0)} max ${t.latency_max?.toFixed(0)} ms · answer age p50 ${t.age_p50?.toFixed(0)} p95 ${t.age_p95?.toFixed(0)} ms · ${(t.tokens/1e6).toFixed(2)} Mtok`;
const c=g('spark').getContext('2d');c.clearRect(0,0,300,40);c.fillStyle='#4a8';const sp=t.sparkline||[];sp.forEach((v,i)=>{const hh=Math.min(40,v/25);c.fillRect(i*5,40-hh,4,hh)});
let e='<tr><th>label</th><th>looks like</th><th>x</th><th>y</th><th>horiz</th><th>bearing</th><th>status</th></tr>';
for(const o of s.entities)e+=`<tr><td class=ch>${o.label}</td><td>${o.description}</td><td>${o.xyz[0].toFixed(3)}</td><td>${o.xyz[1].toFixed(3)}</td><td>${(o.horizontal_m*100).toFixed(0)} cm</td><td>${o.bearing_deg.toFixed(0)}°</td><td>${o.in_view?'in view':'out of view '+o.last_seen_s.toFixed(0)+'s'}${o.reachable?'':' <span class=warn>unreachable</span>'}</td></tr>`;
g('ent').innerHTML=e;g('per').textContent=JSON.stringify(s.perception);
g('ev').textContent=s.events.slice().reverse().map(x=>new Date(x.t*1000).toLocaleTimeString()+' '+x.kind+' '+JSON.stringify(Object.fromEntries(Object.entries(x).filter(([k])=>k!='t'&&k!='kind')))).join('\n');
g('state').textContent=JSON.stringify(s.state,null,1);
if(!seeded){g('task').value=s.task;g('orders').value=s.orders.join('\n');seeded=true}
}catch(e){console.log(e)}}
setInterval(poll,250);setInterval(()=>{g('cam').src='/frame.jpg?'+Date.now();g('third').src='/third.jpg?'+Date.now()},200);poll();
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

    app.add_routes([web.get("/", index), web.get("/api/snapshot", snapshot), web.get("/frame.jpg", frame), web.get("/third.jpg", third),
                    web.post("/api/orders", orders), web.post("/api/task", task), web.post("/api/stop", stop),
                    web.post("/api/resume", resume), web.post("/api/pause", pause), web.post("/api/unpause", unpause),
                    web.post("/api/name", name)])
    return app
