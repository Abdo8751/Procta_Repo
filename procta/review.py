"""Minimal human-review page (first version of the proctor dashboard).

Generates a self-contained static HTML file next to the evidence images. The
invigilator sees each alert with its evidence and confidence, and must Confirm or
Dismiss it. Decisions and the time taken per decision can be exported as CSV - this
is how review effort (RQ2) is measured: baseline alert list vs improved alert list.
"""

import html
import json


def write_review_page(path, alerts, evidence_files, title, sort_by_confidence=True):
    items = []
    for i, (a, img) in enumerate(zip(alerts, evidence_files)):
        items.append({"id": i, "track": a.track_id, "event": a.event_type, "t": round(a.t_alert, 1),
                      "start": round(a.t_start, 1), "conf": None if a.confidence != a.confidence else a.confidence,
                      "img": img, "msg": a.message()})
    if sort_by_confidence:
        items.sort(key=lambda x: -(x["conf"] or 0))
    data = json.dumps(items)
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f4f2;color:#222}}
header{{padding:14px 18px;background:#1f2a36;color:#fff}} header p{{margin:4px 0 0;opacity:.8;font-size:13px}}
main{{max-width:980px;margin:0 auto;padding:16px}}
.card{{background:#fff;border-radius:8px;padding:12px;margin:0 0 14px;box-shadow:0 1px 3px #0002}}
.card img{{width:100%;border-radius:4px}} .meta{{display:flex;gap:14px;flex-wrap:wrap;font-size:14px;margin:8px 0}}
.conf{{font-weight:600}} button{{padding:8px 16px;margin-right:8px;border:0;border-radius:6px;cursor:pointer;font-size:14px}}
.ok{{background:#2e7d32;color:#fff}} .no{{background:#9e9e9e;color:#fff}} .done{{opacity:.45}}
#bar{{position:sticky;top:0;background:#fff;padding:10px 16px;border-bottom:1px solid #ddd;font-size:14px}}
</style></head><body>
<header><b>{html.escape(title)}</b><p>Alerts are prompts for human review only. The system never decides that a student cheated.</p></header>
<div id="bar"><span id="progress"></span> &nbsp; <button class="ok" onclick="exportCsv()">Export decisions CSV</button></div>
<main id="list"></main>
<script>
const items={data}; const t0=performance.now(); let last=t0; const decisions={{}};
function render(){{
 const list=document.getElementById('list'); list.innerHTML='';
 for(const it of items){{
  const d=decisions[it.id]; const c=document.createElement('div'); c.className='card'+(d?' done':'');
  c.innerHTML=`<img src="${{it.img}}" loading="lazy" alt="evidence">
   <p><b>${{it.msg}}</b></p><div class="meta"><span>${{it.event}}</span><span>track ${{it.track}}</span>
   <span>episode ${{it.start}}s → alert ${{it.t}}s</span>
   <span class="conf">confidence ${{it.conf===null?'n/a':it.conf.toFixed(2)}}</span>
   ${{d?`<span>decision: ${{d.decision}} (${{d.secs.toFixed(1)}} s)</span>`:''}}</div>
   <button class="ok">Confirm</button><button class="no">Dismiss</button>`;
  const [b1,b2]=c.querySelectorAll('button'); b1.onclick=()=>decide(it,'confirm'); b2.onclick=()=>decide(it,'dismiss');
  list.appendChild(c);
 }}
 document.getElementById('progress').textContent=`${{Object.keys(decisions).length}} / ${{items.length}} reviewed · ${{((performance.now()-t0)/1000).toFixed(0)}} s total`;
}}
function decide(it,dec){{const now=performance.now(); decisions[it.id]={{decision:dec,secs:(now-last)/1000}}; last=now; render();}}
function exportCsv(){{
 let s='alert_id,track_id,event_type,t_alert,confidence,decision,seconds_to_decide\\n';
 for(const it of items){{const d=decisions[it.id]||{{decision:'',secs:''}}; s+=[it.id,it.track,it.event,it.t,it.conf??'',d.decision,d.secs===''?'':d.secs.toFixed(2)].join(',')+'\\n';}}
 const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([s],{{type:'text/csv'}})); a.download='review_decisions.csv'; a.click();
}}
render();
</script></body></html>"""
    with open(path, "w") as f:
        f.write(page)
