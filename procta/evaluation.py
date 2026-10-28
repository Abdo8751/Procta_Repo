"""Event-level evaluation of alerts against labelled ground-truth episodes.

Ground truth CSV columns: event_type,start_s,end_s[,seat][,note]
Alerts CSV columns:       track_id,event_type,t_alert,t_start,confidence,positive_ratio

Matching rule (per event type):
  * when a GT row has a seat (group sessions: F1..B3, pairs "B2+F2" for exchanges), an
    alert only matches it if the alert's seat is the SAME (pairs compared order-free);
    GT rows without a seat (single-participant sessions) match on type only
  * a GT episode is a TARGET if it lasts >= the event's persistence (3 s; 1 s for
    material_exchange) - shorter ones are "should-not-alert" hard negatives, e.g. a 1 s
    glance at the phone; lookalike_* rows are never targets
  * an alert is a TRUE alert if t_alert falls in [start, end + tolerance] of a
    TARGET episode of the same type, otherwise it is a FALSE alarm
  * an episode is DETECTED if it has >= 1 true alert

Metrics:
  precision        = true alerts / all alerts
  recall           = detected target episodes / target episodes
  missed-event rate= 1 - recall
  alerts per exam  = "alerts" column (total alerts the invigilator must review)
  false alarms/min = false alerts / session minutes
  latency          = first true alert time - episode start   (>= persistence_s by design)
  alerts/episode   = true alerts / detected episodes  (1.0 ideal; >1 = duplicate
                     alerts a human must review -> RQ2 "review effort")
"""

import csv
import statistics as st
from collections import defaultdict


def norm_seat(seat):
    """'f2+B2' -> 'B2+F2'; '' / None -> None."""
    if seat is None or isinstance(seat, int):
        return None
    parts = sorted(p.strip().upper() for p in str(seat).split("+") if p.strip())
    return "+".join(parts) or None


def load_gt(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"event_type": r["event_type"].strip(), "start_s": float(r["start_s"]),
                         "end_s": float(r["end_s"]), "seat": norm_seat(r.get("seat"))})
    return rows


def alert_rows(alerts):
    """temporal.Alert list -> rows for evaluate(); the seat is the identity when it is a seat label."""
    return [{"event_type": a.event_type, "t_alert": a.t_alert,
             "seat": norm_seat(a.track_id) if isinstance(a.track_id, str) else None} for a in alerts]


def load_alerts(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"event_type": r["event_type"], "t_alert": float(r["t_alert"]),
                         "confidence": float(r.get("confidence") or "nan")})
    return rows


def evaluate(alerts, gt, duration_s, persistence_s=3.0, tolerance_s=1.0, event_types=None,
             persistence_by_event=None):
    """persistence_by_event: {event_type: seconds} overriding persistence_s (e.g. Config.event_persistence_s)."""
    types = sorted(event_types or {g["event_type"] for g in gt} | {a["event_type"] for a in alerts})
    types = [t for t in types if not t.startswith("lookalike")]
    pers = persistence_by_event or {}
    per = {}
    tot = defaultdict(float)
    all_lat = []
    for ev in types:
        p_ev = pers.get(ev, persistence_s)
        targets = [g for g in gt if g["event_type"] == ev and g["end_s"] - g["start_s"] >= p_ev]
        hits = defaultdict(list)
        n_true = n_false = 0
        for a in (a for a in alerts if a["event_type"] == ev):
            a_seat = norm_seat(a.get("seat"))
            m = next((i for i, g in enumerate(targets)
                      if g["start_s"] <= a["t_alert"] <= g["end_s"] + tolerance_s
                      and (not g.get("seat") or g["seat"] == a_seat)), None)
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
            "missed_event_rate": 1 - detected / len(targets) if targets else float("nan"),
            "false_alarms_per_min": n_false / (duration_s / 60.0),
            "false_alerts_per_hour": n_false / (duration_s / 3600.0),
            "duplicate_alerts": n_true - detected,
            "mean_latency_s": st.mean(lat) if lat else float("nan"),
            "alerts_per_episode": n_true / detected if detected else float("nan"),
        }
        for k in ("alerts", "true_alerts", "false_alerts", "target_episodes", "detected"):
            tot[k] += per[ev][k]
    per["ALL"] = {
        **{k: int(v) for k, v in tot.items()},
        "precision": tot["true_alerts"] / tot["alerts"] if tot["alerts"] else float("nan"),
        "recall": tot["detected"] / tot["target_episodes"] if tot["target_episodes"] else float("nan"),
        "missed_event_rate": (1 - tot["detected"] / tot["target_episodes"]) if tot["target_episodes"] else float("nan"),
        "false_alarms_per_min": tot["false_alerts"] / (duration_s / 60.0),
        "false_alerts_per_hour": tot["false_alerts"] / (duration_s / 3600.0),
        "duplicate_alerts": int(tot["true_alerts"] - tot["detected"]),
        "mean_latency_s": st.mean(all_lat) if all_lat else float("nan"),
        "alerts_per_episode": tot["true_alerts"] / tot["detected"] if tot["detected"] else float("nan"),
    }
    return per


def evaluate_per_seat(alerts, gt, duration_s, persistence_s=3.0, tolerance_s=1.0, event_types=None,
                      persistence_by_event=None):
    """{seat: evaluate(...)} over the seats that appear in the GT or the alerts (pairs as their own key)."""
    seats = sorted({g["seat"] for g in gt if g.get("seat")} | {norm_seat(a.get("seat")) for a in alerts
                                                                 if norm_seat(a.get("seat"))})
    return {s: evaluate([a for a in alerts if norm_seat(a.get("seat")) == s], [g for g in gt if g.get("seat") == s],
                        duration_s, persistence_s, tolerance_s, event_types, persistence_by_event)
            for s in seats}


COLS = ["alerts", "target_episodes", "detected", "precision", "recall", "missed_event_rate",
        "false_alerts_per_hour", "duplicate_alerts", "mean_latency_s", "alerts_per_episode"]


def aggregate(results_list, durations):
    """Combine per-session results (from evaluate) into one result over all sessions:
    counts are summed, rates recomputed, latency averaged over detected episodes."""
    total_s = sum(durations)
    types = sorted({ev for r in results_list for ev in r if ev != "ALL"}) + ["ALL"]
    out = {}
    for ev in types:
        c = {k: sum(r.get(ev, {}).get(k, 0) for r in results_list)
             for k in ("alerts", "true_alerts", "false_alerts", "target_episodes", "detected")}
        lat_w = [(r[ev]["mean_latency_s"], r[ev]["detected"]) for r in results_list
                 if ev in r and r[ev]["detected"] and r[ev]["mean_latency_s"] == r[ev]["mean_latency_s"]]
        out[ev] = {
            **c,
            "precision": c["true_alerts"] / c["alerts"] if c["alerts"] else float("nan"),
            "recall": c["detected"] / c["target_episodes"] if c["target_episodes"] else float("nan"),
            "missed_event_rate": 1 - c["detected"] / c["target_episodes"] if c["target_episodes"] else float("nan"),
            "false_alarms_per_min": c["false_alerts"] / (total_s / 60.0),
            "false_alerts_per_hour": c["false_alerts"] / (total_s / 3600.0),
            "duplicate_alerts": c["true_alerts"] - c["detected"],
            "mean_latency_s": (sum(l * n for l, n in lat_w) / sum(n for _, n in lat_w)) if lat_w else float("nan"),
            "alerts_per_episode": c["true_alerts"] / c["detected"] if c["detected"] else float("nan"),
        }
    return out


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
