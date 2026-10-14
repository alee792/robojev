"""Per-run summary of the 2026-10-12/13 live slice recordings (experiments/results/live/slice_*2026101[23]-*.jsonl.gz):
user-text decisions with confidence, plans sent back, readings with a transient phrase, reading_kept notes.

    uv run --frozen python experiments/live_slice_rerun_summary.py
"""
import json,glob,re,sys,collections,gzip
from robojev.brain.rules import TRANSIENT_RE
rows=[]; tot=collections.defaultdict(collections.Counter)
for f in sorted(glob.glob("experiments/results/live/slice_*2026101[23]-*.jsonl.gz")):
    m=re.match(r".*slice_(\w+)-openai_test_(\w)_.*_(\d+-\d+)\.jsonl.gz",f); dec,sc,st=m.groups()
    log=[];decs=[];done=0;lat=[]
    for l in gzip.open(f,"rt"):
        r=json.loads(l)
        if r.get("type")=="event" and r.get("kind")=="skill_done": done+=1
        if r.get("type")!="note": continue
        if r["kind"]=="brain_log": log.append(r["rec"])
        if r["kind"]=="decision": decs.append(r["rec"]); lat.append(r["rec"].get("latency_ms"))
    if not log: continue
    ut=[d for d in decs if d.get("event")=="user_text"]
    pa=[d for d in decs if d.get("event")=="plan_arrived" and not d["text"].startswith("the first")]
    sent=sum(d["combined"]["route"]!="stay_local" for d in pa)
    readings=[(x.get("plan") or x.get("diff") or {}).get("reading","") for x in log if x["note"]=="plan_applied"]
    leak=[x for x in readings if TRANSIENT_RE.search(x or "")]
    kept=sum(x["note"]=="reading_kept" for x in log)
    esc=sum(d["combined"]["route"]!="stay_local" for d in decs)
    end=[x.get("outcome") for x in log if x["note"]=="end"]
    t=collections.Counter(tot[(dec,sc)]); 
    tot[dec].update(new_plans=len(pa),sent=sent,decisions=len(decs),escalated=esc,llm=sum(x["note"]=="llm_request" for x in log))
    print(dec,sc,st,end,"steps",done,"| user_text:",[(d["combined"]["right_now"],d["combined"]["conf"]["right_now"],d["combined"]["route"]) for d in ut],
          "| plans sent back",sent,"/",len(pa),"| llm",sum(x["note"]=="llm_request" for x in log),"| leak",leak,"kept",kept,"| final reading",readings[-1][:60] if readings else None, "| lat max", max(l for l in lat if l))
for k,v in tot.items(): print(k,dict(v))
