"""Local alert dashboard: every live alert from results/alerts.db, updating while
live.py runs, with Confirm / Dismiss review buttons that are saved to the database.

Priority: a student (seat or track, per session) with repeated suspicious activity is
flagged higher - activity = alerts plus repeated sightings merged into them; dismissed alerts
do not count; an exchange alert counts for both students. 2 = Elevated, 3 or more = High.

  python scripts/dashboard.py                # then open http://localhost:8765
  python scripts/dashboard.py --port 9000

Only listens on 127.0.0.1: evidence images show real people and never leave this PC.
Older results/live_* sessions (from before the log existed) are imported on start.
"""

import argparse
import csv
import io
import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from procta.alert_log import connect, delete_alert, delete_session, import_live_dirs, purge_evidence  # noqa: E402

RESULTS = ROOT / "results"
DB = RESULTS / "alerts.db"
CSV_COLS = ["id", "session", "mode", "timestamp", "t_session_s", "track_id", "event_type", "alert_name",
            "message", "confidence", "confidence_level", "positive_ratio", "episode_start_s", "duration_s",
            "evidence", "observation_point", "sightings", "last_seen_s", "clips", "region", "context_note",
            "review", "reviewed_at"]
MEDIA = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif", ".mp4": "video/mp4"}


def query_alerts(session=None):
    con = connect(DB)
    sql, args = "SELECT * FROM alerts", ()
    if session:
        sql, args = sql + " WHERE session=?", (session,)
    rows = [dict(r) for r in con.execute(sql + " ORDER BY timestamp DESC, id DESC", args)]
    con.close()
    return rows


def query_observations(session=None):
    con = connect(DB)
    sql, args = "SELECT * FROM observations", ()
    if session:
        sql, args = sql + " WHERE session=?", (session,)
    rows = [dict(r) for r in con.execute(sql + " ORDER BY timestamp DESC, id DESC LIMIT 200", args)]
    con.close()
    return rows


def query_sessions():
    con = connect(DB)
    rows = [dict(r) for r in con.execute(
        "SELECT s.session, s.mode, s.started_at, s.ended_at, s.resolution, COUNT(a.id) AS n_alerts"
        " FROM sessions s LEFT JOIN alerts a ON a.session = s.session WHERE NOT s.deleted"
        " GROUP BY s.session ORDER BY s.started_at DESC")]
    con.close()
    for r in rows:
        r.pop("config_json", None)
    return rows


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200, extra=None):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            return self._send(PAGE, "text/html; charset=utf-8")
        if url.path == "/api/alerts":
            return self._send(json.dumps(query_alerts(q.get("session"))))
        if url.path == "/api/observations":
            return self._send(json.dumps(query_observations(q.get("session"))))
        if url.path == "/api/sessions":
            return self._send(json.dumps(query_sessions()))
        if url.path == "/api/export.csv":
            buf = io.StringIO()
            w = csv.DictWriter(buf, CSV_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(query_alerts(q.get("session")))
            name = f"procta_alerts_{q.get('session') or 'all'}.csv"
            return self._send(buf.getvalue(), "text/csv; charset=utf-8",
                              extra={"Content-Disposition": f'attachment; filename="{name}"'})
        if url.path.startswith("/evidence/"):
            f = (RESULTS / url.path[len("/evidence/"):]).resolve()
            if RESULTS.resolve() in f.parents and f.suffix.lower() in MEDIA and f.is_file():
                return self._send(f.read_bytes(), MEDIA[f.suffix.lower()])
            return self._send(b"not found", "text/plain", 404)
        return self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if path in ("/api/delete", "/api/delete_session"):
            con = connect(DB)
            if path == "/api/delete":
                purge_evidence(con, body["id"], RESULTS)
            n = delete_alert(con, body["id"]) if path == "/api/delete" else delete_session(con, body["session"])
            con.close()
            return self._send(json.dumps({"ok": True, "deleted": n}))
        if path != "/api/review":
            return self._send(b"not found", "text/plain", 404)
        decision = body.get("review")
        if decision not in ("confirmed", "dismissed", None):
            return self._send(json.dumps({"error": "bad review"}), code=400)
        con = connect(DB)
        with con:
            con.execute("UPDATE alerts SET review=?, reviewed_at=? WHERE id=?",
                        (decision, datetime.now().isoformat(timespec="seconds") if decision else None,
                         int(body["id"])))
        purged = purge_evidence(con, body["id"], RESULTS) if decision == "dismissed" else 0
        con.close()
        return self._send(json.dumps({"ok": True, "files_deleted": purged}))

    def log_message(self, *args):          # keep the terminal quiet
        pass


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PROCTA Alerts</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--text:#1d2430;--muted:#667085;--line:#e3e6ea;--accent:#1f4e8c;
 --high:#1e7b3a;--med:#b26a00;--low:#b42318;--ok:#1e7b3a;--no:#6b7280}
@media (prefers-color-scheme:dark){:root{--bg:#12161c;--card:#1b2129;--text:#e6e9ee;--muted:#98a2b3;
 --line:#2c3440;--accent:#7fb0f0;--high:#4cc274;--med:#f0a640;--low:#f07167;--ok:#4cc274;--no:#98a2b3}}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:12px;align-items:center}
header h1{font-size:18px;margin:0 12px 0 0} header .note{color:var(--muted);font-size:12px;flex-basis:100%}
select,button,a.btn{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--text);cursor:pointer;text-decoration:none}
main{padding:16px 20px;max-width:1300px;margin:0 auto}
.live{display:inline-flex;align-items:center;gap:6px;color:var(--muted)} .dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}
.dot.on{background:var(--low);animation:p 1.2s infinite} @keyframes p{50%{opacity:.3}}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card .k{color:var(--muted);font-size:12px} .card .v{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}
h2{font-size:15px;margin:18px 0 8px}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;vertical-align:middle}
th{font-size:12px;color:var(--muted);font-weight:600} tr:last-child td{border-bottom:0}
tr.new td{animation:hl 3s} @keyframes hl{from{background:rgba(255,200,0,.25)}}
.bar{display:inline-block;width:70px;height:7px;border-radius:4px;background:var(--line);margin-right:6px;vertical-align:middle;overflow:hidden}
.bar i{display:block;height:100%} .High{color:var(--high)} .Medium{color:var(--med)} .Low{color:var(--low)}
.bar .High{background:var(--high)} .bar .Medium{background:var(--med)} .bar .Low{background:var(--low)}
img.thumb{height:44px;border-radius:4px;cursor:zoom-in;display:block}
button.del{border-color:var(--low);color:var(--low)} button.ok{border-color:var(--ok);color:var(--ok)} button.no{border-color:var(--no);color:var(--no)}
.tag{padding:2px 8px;border-radius:10px;font-size:12px;border:1px solid currentColor}
.confirmed{color:var(--ok)} .dismissed{color:var(--no)}
.prio{padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;border:1px solid currentColor;white-space:nowrap}
.prio.High{color:var(--low)} .prio.Elevated{color:var(--med)} .prio.Normal{color:var(--muted);font-weight:400}
tr.hi td:first-child{box-shadow:inset 4px 0 0 var(--low)} tr.el td:first-child{box-shadow:inset 4px 0 0 var(--med)}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;align-items:center;justify-content:center;padding:16px}
#modal img,#modal video{max-width:100%;max-height:100%} th.grp{text-align:left;color:var(--text);border-bottom:2px solid var(--line)}
th.human,td.human{border-left:2px solid var(--accent)} a.clip{margin-right:6px;font-size:12px} .muted{color:var(--muted)} .empty{padding:24px;text-align:center;color:var(--muted)}
</style></head><body>
<header><h1>PROCTA alert log</h1>
 <label>Session <select id="session"><option value="">All sessions</option></select></label>
 <label>Alert <select id="type"><option value="">All types</option></select></label>
 <a class="btn" id="csv" href="/api/export.csv">Download CSV</a>
 <button class="del" id="delsess" style="display:none" onclick="delSession()">Delete session</button>
 <label><input type="checkbox" id="showdis"> Show dismissed (<span id="ndis">0</span>)</label>
 <label><input type="checkbox" id="byprio"> Sort by priority</label>
 <span class="live"><span class="dot" id="dot"></span><span id="livetxt">idle</span></span>
 <div class="note">Alerts are prompts for human review only &mdash; the system never decides that a student cheated.
 Confidence = detector confidence &times; how consistently the behaviour was seen. Accuracy = confirmed &divide; reviewed.
 Dismissing an alert deletes its still and clips from this PC and hides it; the decision is kept so accuracy stays correct.</div>
</header>
<main>
 <div class="cards">
  <div class="card"><div class="k">Alerts</div><div class="v" id="c_total">0</div></div>
  <div class="card"><div class="k">Reviewed</div><div class="v" id="c_rev">0</div></div>
  <div class="card"><div class="k">Accuracy (confirmed / reviewed)</div><div class="v" id="c_acc">&ndash;</div></div>
  <div class="card"><div class="k">Average confidence</div><div class="v" id="c_conf">&ndash;</div></div>
  <div class="card"><div class="k">Alerts per minute (selected sessions)</div><div class="v" id="c_rate">&ndash;</div></div>
 </div>
 <h2>Students needing attention</h2>
 <p class="muted" style="margin:0 0 8px;font-size:13px">Repeated suspicious activity by the same student in one session: alerts plus repeated sightings, dismissed alerts excluded. 2 = Elevated, 3 or more = High.</p>
 <div class="wrap"><table><thead><tr><th>Priority</th><th>Seat / track</th><th>Session</th><th>Activity count</th><th>Alerts</th><th>What was seen</th><th>Last alert</th></tr></thead><tbody id="prio"></tbody></table></div>
 <h2>By alert type</h2>
 <div class="wrap"><table><thead><tr><th>Alert</th><th>Count</th><th>Avg confidence</th><th>Confirmed</th><th>Dismissed</th><th>Not reviewed</th><th>Accuracy</th></tr></thead><tbody id="types"></tbody></table></div>
 <h2>Alerts (newest first)</h2>
 <div class="wrap"><table><thead>
  <tr><th colspan="12" class="grp">Automatic detection (what the system observed)</th><th class="grp human">Invigilator decision</th><th class="grp"></th></tr>
  <tr><th>Time</th><th>Session time</th><th>Point</th><th>Event</th><th>Seat / track</th><th>Priority</th><th>Confidence</th>
  <th>Consistency</th><th>Lasted before alert</th><th>Sightings</th><th>Mode</th><th>Evidence</th><th class="human">Review</th><th></th></tr></thead><tbody id="rows"></tbody></table></div>
 <h2>Robot observation log</h2>
 <div class="wrap"><table><thead><tr><th>Time</th><th>Session time</th><th>Point</th><th>What happened</th><th>Note</th></tr></thead><tbody id="obs"></tbody></table></div>
</main>
<div id="modal" onclick="this.style.display='none'"><img id="big" alt="evidence"></div>
<script>
const $=id=>document.getElementById(id); let alerts=[], sessions=[], obs=[], seen=null, sig='';
const pct=x=>x==null?'–':(100*x).toFixed(0)+'%', num=(x,d=2)=>x==null?'–':x.toFixed(d);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const PRIO_ELEVATED=2, PRIO_HIGH=3, RANK={High:2,Elevated:1,Normal:0};
const level=n=>n>=PRIO_HIGH?'High':n>=PRIO_ELEVATED?'Elevated':'Normal';
const people=a=>String(a.track_id).split('+').filter(p=>p&&p!=='INV');
function priorities(all){            // session|person -> repeated-activity summary
 const m={};
 for(const a of all){ if(a.review==='dismissed') continue;
  for(const p of people(a)){ const e=m[a.session+'|'+p]??={session:a.session,who:p,n:0,alerts:0,types:{},last:''};
   e.n+=(a.sightings??1); e.alerts++; e.types[a.alert_name]=(e.types[a.alert_name]||0)+1; if(a.timestamp>e.last)e.last=a.timestamp; } }
 return m;
}
function alertPrio(a,pm){ let best='Normal', n=0;
 for(const p of people(a)){ const e=pm[a.session+'|'+p]; if(e&&RANK[level(e.n)]>=RANK[best]){best=level(e.n); n=Math.max(n,e.n);} }
 return {lvl:best,n}; }
async function load(){
 const s=$('session').value;
 const qs=s?'?session='+encodeURIComponent(s):'';
 const [a,ss,ob]=await Promise.all([fetch('/api/alerts'+qs).then(r=>r.json()),
                                 fetch('/api/sessions').then(r=>r.json()), fetch('/api/observations'+qs).then(r=>r.json())]);
 const nsig=JSON.stringify([a.map(x=>[x.id,x.review,x.sightings,x.clips]),ss.map(x=>[x.session,x.ended_at,x.n_alerts]),ob.length]);
 if(nsig===sig) return; sig=nsig; alerts=a; sessions=ss; obs=ob; render();
}
function render(){
 const sel=$('session'), cur=sel.value;
 sel.innerHTML='<option value="">All sessions</option>'+sessions.map(x=>`<option value="${esc(x.session)}">${esc(x.session)} (${x.n_alerts})${x.ended_at===null?' ● live':''}</option>`).join('');
 sel.value=cur;
 const running=sessions.filter(x=>x.ended_at===null);
 $('dot').className='dot'+(running.length?' on':''); $('livetxt').textContent=running.length?'live session running':'no live session';
 $('delsess').style.display=cur?'':'none';
 $('csv').href='/api/export.csv'+(cur?'?session='+encodeURIComponent(cur):'');
 const tsel=$('type'), tcur=tsel.value, names=[...new Set(alerts.map(a=>a.alert_name))].sort();
 tsel.innerHTML='<option value="">All types</option>'+names.map(n=>`<option>${esc(n)}</option>`).join(''); tsel.value=tcur;
 const list=alerts.filter(a=>!tsel.value||a.alert_name===tsel.value);
 const rev=list.filter(a=>a.review), conf=list.filter(a=>a.confidence!=null);
 $('c_total').textContent=list.length; $('c_rev').textContent=rev.length;
 $('c_acc').textContent=rev.length?pct(rev.filter(a=>a.review==='confirmed').length/rev.length):'–';
 $('c_conf').textContent=conf.length?num(conf.reduce((s,a)=>s+a.confidence,0)/conf.length):'–';
 const used=sessions.filter(x=>!cur||x.session===cur); let mins=0;
 for(const x of used){const sa=alerts.filter(a=>a.session===x.session); const end=x.ended_at?new Date(x.ended_at):(x.ended_at===null?new Date():null);
  mins+=end?Math.max(0,(end-new Date(x.started_at))/60000):Math.max(0,...sa.map(a=>a.t_session_s))/60;}
 $('c_rate').textContent=mins>0.05?num(list.length/mins):'–';
 const by={}; for(const a of list){const t=by[a.alert_name]??={n:0,c:0,cn:0,ok:0,no:0}; t.n++; if(a.confidence!=null){t.c+=a.confidence;t.cn++}
  if(a.review==='confirmed')t.ok++; if(a.review==='dismissed')t.no++;}
 $('types').innerHTML=Object.entries(by).map(([n,t])=>`<tr><td>${esc(n)}</td><td>${t.n}</td><td>${t.cn?num(t.c/t.cn):'–'}</td><td>${t.ok}</td><td>${t.no}</td><td>${t.n-t.ok-t.no}</td><td>${t.ok+t.no?pct(t.ok/(t.ok+t.no)):'–'}</td></tr>`).join('')
  ||'<tr><td colspan="7" class="empty">No alerts yet</td></tr>';
 const ids=new Set(alerts.map(a=>a.id));
 const ndis=list.filter(a=>a.review==='dismissed').length; $('ndis').textContent=ndis;
 const pm=priorities(alerts);
 const att=Object.values(pm).filter(e=>e.n>=PRIO_ELEVATED).sort((x,y)=>y.n-x.n||(y.last>x.last?1:-1));
 $('prio').innerHTML=att.map(e=>`<tr class="${level(e.n)==='High'?'hi':'el'}"><td><span class="prio ${level(e.n)}">${level(e.n)}</span></td><td><b>${esc(e.who)}</b></td>
  <td class="muted">${esc(e.session)}</td><td>${e.n}</td><td>${e.alerts}</td>
  <td>${Object.entries(e.types).map(([k,v])=>esc(k)+(v>1?' &times;'+v:'')).join(', ')}</td><td>${esc(e.last.replace('T',' '))}</td></tr>`).join('')
  ||'<tr><td colspan="7" class="empty">No student with repeated activity</td></tr>';
 let shown=$('showdis').checked?list:list.filter(a=>a.review!=='dismissed');
 if($('byprio').checked) shown=[...shown].sort((x,y)=>{const px=alertPrio(x,pm), py=alertPrio(y,pm);
  return RANK[py.lvl]-RANK[px.lvl]||py.n-px.n||(y.timestamp>x.timestamp?1:-1);});
 $('rows').innerHTML=shown.map(a=>{const lvl=a.confidence_level, isNew=seen&&!seen.has(a.id), pr=alertPrio(a,pm);
  const clips=JSON.parse(a.clips||'[]');
  return `<tr class="${isNew?'new':''} ${pr.lvl==='High'?'hi':pr.lvl==='Elevated'?'el':''}"><td>${esc(a.timestamp.replace('T',' '))}</td><td>${num(a.t_session_s,1)} s</td>
  <td>${esc(a.observation_point||'–')}</td>
  <td title="${esc(a.message)}${a.context_note?'\nContext: '+esc(a.context_note):''}"><b>${esc(a.alert_name)}</b></td><td>${a.track_id}</td>
  <td><span class="prio ${pr.lvl}" title="${pr.n} suspicious observation(s) for this student in this session">${pr.lvl}${pr.n>1?' &middot; '+pr.n:''}</span></td>
  <td><span class="bar"><i class="${lvl}" style="width:${(a.confidence??0)*100}%"></i></span><span class="${lvl}">${num(a.confidence)} ${lvl}</span></td>
  <td>${pct(a.positive_ratio)}</td><td>${num(a.duration_s,1)} s</td><td>${a.sightings??1}${(a.sightings??1)>1?' (consolidated)':''}</td><td>${esc(a.mode)}</td>
  <td>${a.evidence?`<img class="thumb" loading="lazy" src="/evidence/${encodeURI(a.evidence)}" onclick="zoom(this.src)" alt="evidence">`:`<span class="muted">${a.review==='dismissed'?'deleted':'none'}</span>`}
   ${clips.map((c,i)=>`<a class="clip" href="#" onclick="zoom('/evidence/${encodeURI(c)}');return false">clip ${i+1}</a>`).join('')}</td>
  <td class="human">${a.review?`<span class="tag ${a.review}">${a.review}</span> <button onclick="review(${a.id},null)">undo</button>`
   :`<button class="ok" onclick="review(${a.id},'confirmed')">Confirm</button> <button class="no" onclick="review(${a.id},'dismissed')">Dismiss</button>`}</td>
  <td><button class="del" title="Delete this alert" onclick="delAlert(${a.id})">Delete</button></td></tr>`}).join('')
  ||'<tr><td colspan="14" class="empty">No alerts yet &mdash; run python scripts/live.py</td></tr>';
 const kinds={arrive:'Arrived at point',leave:'Left point',insufficient_view:'Insufficient view - asked to move',
  permitted_device:'Permitted device seen (logged, not alerted)',authorised_exchange:'Authorised exchange with invigilator (logged, not alerted)'};
 $('obs').innerHTML=obs.map(o=>`<tr><td>${esc(o.timestamp.replace('T',' '))}</td><td>${num(o.t_session_s,1)} s</td><td>${esc(o.point)}</td><td>${esc(kinds[o.kind]||o.kind)}</td><td class="muted">${esc(o.note)}</td></tr>`).join('')
  ||'<tr><td colspan="5" class="empty">No robot observations yet</td></tr>';
 seen=ids;
}
async function review(id,r){await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id,review:r})}); await load();}
async function post(url,body){await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); sig=''; await load();}
function delAlert(id){const a=alerts.find(x=>x.id===id);
 if(confirm(`Delete this alert?

${a.alert_name} at ${a.timestamp.replace('T',' ')}

This cannot be undone.`)) post('/api/delete',{id});}
function delSession(){const s=$('session').value, n=alerts.length;
 if(s&&confirm(`Delete session ${s} and all ${n} of its alerts from the log?

This cannot be undone.`)){$('session').value=''; post('/api/delete_session',{session:s});}}
function zoom(src){$('big').src=src+(src.endsWith('.webp')||src.endsWith('.gif')?'?'+Date.now():''); $('modal').style.display='flex';}
$('session').onchange=()=>{sig='';seen=null;load()}; $('type').onchange=render; $('showdis').onchange=render; $('byprio').onchange=render;
load(); setInterval(load,2000);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    added = import_live_dirs(DB, RESULTS)
    if added:
        print(f"Imported {len(added)} older live session(s): {', '.join(added)}")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"PROCTA dashboard -> http://localhost:{args.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
