"""Event-level evaluation of alerts against labelled ground-truth episodes.

Ground truth CSV columns: event_type,start_s,end_s[,note]
Alerts CSV columns:       track_id,event_type,t_alert,t_start,confidence,positive_ratio

Matching rule (per event type, track ids ignored because staged sessions have one
student per camera):
  * a GT episode is a TARGET if it lasts >= persistence_s (shorter ones are
    "should-not-alert" hard negatives, e.g. a 1 s glance at the phone)
  * an alert is a TRUE alert if t_alert falls in [start, end + tolerance] of a
    TARGET episode of the same type, otherwise it is a FALSE alarm
  * an episode is DETECTED if it has >= 1 true alert

Metrics:
  precision        = true alerts / all alerts
  recall           = detected target episodes / target episodes
  false alarms/min = false alerts / session minutes
  latency          = first true alert time - episode start   (>= persistence_s by design)
  alerts/episode   = true alerts / detected episodes  (1.0 ideal; >1 = duplicate
                     alerts a human must review -> RQ2 "review effort")
"""

import csv
import statistics as st
from collections import defaultdict


def load_gt(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"event_type": r["event_type"].strip(),
                         "start_s": float(r["start_s"]), "end_s": float(r["end_s"])})
    return rows


def load_alerts(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"event_type": r["event_type"], "t_alert": float(r["t_alert"]),
                         "confidence": float(r.get("confidence") or "nan")})
    return rows


def evaluate(alerts, gt, duration_s, persistence_s=3.0, tolerance_s=1.0, event_types=None):
    types = sorted(event_types or {g["event_type"] for g in gt} | {a["event_type"] for a in alerts})
    per = {}
    tot = defaultdict(float)
    all_lat = []
    for ev in types:
        targets = [g for g in gt if g["event_type"] == ev and g["end_s"] - g["start_s"] >= persistence_s]
        hits = defaultdict(list)
        n_true = n_false = 0
        for a in (a for a in alerts if a["event_type"] == ev):
            m = next((i for i, g in enumerate(targets)
                      if g["start_s"] <= a["t_alert"] <= g["end_s"] + tolerance_s), None)
            if m is None:
                n_false += 1
            else:
                n_true += 1
                hits[m].append(a["t_alert"])
        detected = len(hits)
        lat = [min(ts) - targets[i]["start_s"] for i, ts in hits.items()]
        all_lat += lat
        n_alerts = n_true + n_false
        per[ev] = {
            "alerts": n_alerts, "true_alerts": n_true, "false_alerts": n_false,
            "target_episodes": len(targets), "detected": detected,
            "precision": n_true / n_alerts if n_alerts else float("nan"),
            "recall": detected / len(targets) if targets else float("nan"),
            "false_alarms_per_min": n_false / (duration_s / 60.0),
            "mean_latency_s": st.mean(lat) if lat else float("nan"),
            "alerts_per_episode": n_true / detected if detected else float("nan"),
        }
        for k in ("alerts", "true_alerts", "false_alerts", "target_episodes", "detected"):
            tot[k] += per[ev][k]
    per["ALL"] = {
        **{k: int(v) for k, v in tot.items()},
        "precision": tot["true_alerts"] / tot["alerts"] if tot["alerts"] else float("nan"),
        "recall": tot["detected"] / tot["target_episodes"] if tot["target_episodes"] else float("nan"),
        "false_alarms_per_min": tot["false_alerts"] / (duration_s / 60.0),
        "mean_latency_s": st.mean(all_lat) if all_lat else float("nan"),
        "alerts_per_episode": tot["true_alerts"] / tot["detected"] if tot["detected"] else float("nan"),
    }
    return per


COLS = ["alerts", "target_episodes", "detected", "precision", "recall",
        "false_alarms_per_min", "mean_latency_s", "alerts_per_episode"]


def to_markdown(results: dict, title=""):
    lines = [f"### {title}"] if title else []
    lines.append("| event | " + " | ".join(COLS) + " |")
    lines.append("|---" * (len(COLS) + 1) + "|")
    for ev, r in results.items():
        cells = []
        for c in COLS:
            v = r.get(c, "")
            cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append(f"| {ev} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
