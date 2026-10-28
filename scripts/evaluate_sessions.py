"""Evaluate all four methods over complete sessions, with a development / test split.

Rules from the supervisors (Oct 2026):
  * separate participants or recording sessions between development and testing
  * keep the final decision thresholds FIXED before looking at the test sessions
  * report per category: event-level precision and recall, false alerts per recording
    hour, detection delay, duplicate alerts (+ reviewer effort from the dashboard)

Workflow
  1. List sessions in data/splits.csv:    session,split,point
                                          s01,dev,A
                                          s04,test,B
  2. Tune on dev only:     python scripts/evaluate_sessions.py --set dev
     (change thresholds in procta/config.py or a JSON file, re-run, repeat)
  3. Freeze:               python scripts/evaluate_sessions.py --freeze configs/thresholds_frozen.json
  4. Test ONCE:            python scripts/evaluate_sessions.py --set test --thresholds configs/thresholds_frozen.json

Each session needs data/detections/<session>.jsonl (extract_detections.py) and
data/labels/<session>_gt.csv. Output: results/eval_<set>/summary.md + summary.json.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from procta.config import EVENT_TYPES, load_config  # noqa: E402
from procta.context import ExamContext  # noqa: E402
from procta.detect import load_detections  # noqa: E402
from procta.evaluation import COLS, aggregate, alert_rows, evaluate, load_gt  # noqa: E402
from procta.temporal import MODES, run_policy  # noqa: E402

NAMES = {"frame": "B0 frame-level", "baseline": "B1 3-s persistence",
         "improved": "Temporal + consolidation", "context": "Temporal + consolidation + context"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(ROOT / "data/splits.csv"))
    ap.add_argument("--set", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--context", default=None)
    ap.add_argument("--freeze", default=None, help="write the current thresholds to this JSON and exit")
    ap.add_argument("--tolerance", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config(args.thresholds)
    if args.freeze:
        Path(args.freeze).parent.mkdir(parents=True, exist_ok=True)
        Path(args.freeze).write_text(json.dumps({**cfg.to_dict(), "_frozen_at": datetime.now().isoformat(
            timespec="seconds")}, indent=1))
        print(f"Thresholds frozen -> {args.freeze}. Do not change them before the test run.")
        return
    if args.set == "test" and not args.thresholds:
        sys.exit("Test runs must use frozen thresholds: add --thresholds configs/thresholds_frozen.json")

    ctx = ExamContext.load(args.context) if args.context else ExamContext.load()
    rows = [r for r in csv.DictReader(open(args.splits)) if args.set == "all" or r["split"].strip() == args.set]
    if not rows:
        sys.exit(f"No sessions with split={args.set} in {args.splits}")

    per_mode = {m: [] for m in MODES}
    durations, used, suppressed = [], [], {}
    for r in rows:
        s = r["session"].strip()
        det_f, gt_f = ROOT / "data/detections" / f"{s}.jsonl", ROOT / "data/labels" / f"{s}_gt.csv"
        if not det_f.exists() or not gt_f.exists():
            print(f"skip {s}: missing {'detections' if not det_f.exists() else 'labels'}")
            continue
        meta, frames = load_detections(det_f)
        dur = meta.get("duration_s") or frames[-1]["t"]
        gt = load_gt(gt_f)
        point = ctx.point(r.get("point", "").strip()) if r.get("point") else None
        for m in MODES:
            alerts, _, sig = run_policy(frames, m, cfg, context=ctx, point=point, return_events=True)
            per_mode[m].append(evaluate(alert_rows(alerts),
                                        gt, dur, cfg.persistence_s, args.tolerance, EVENT_TYPES,
                                        cfg.event_persistence_s))
            if m == "context":
                for k, v in sig.suppressed.items():
                    suppressed[k] = suppressed.get(k, 0) + v
        durations.append(dur)
        used.append(s)

    out = ROOT / "results" / f"eval_{args.set}"
    out.mkdir(parents=True, exist_ok=True)
    hours = sum(durations) / 3600
    md = [f"# Evaluation on {args.set} sessions: {', '.join(used)} ({hours * 60:.1f} min total)\n",
          f"Thresholds: {args.thresholds or 'procta/config.py defaults (development)'}\n"]
    summary = {}
    for m in MODES:
        res = aggregate(per_mode[m], durations)
        summary[m] = res
        md.append(f"### {NAMES[m]} (`{m}`)\n")
        md.append("| event | " + " | ".join(COLS) + " |")
        md.append("|---" * (len(COLS) + 1) + "|")
        for ev, v in res.items():
            md.append(f"| {ev} | " + " | ".join(f"{v[c]:.2f}" if isinstance(v[c], float) else str(v[c])
                                                for c in COLS) + " |")
        md.append("")
    if suppressed:
        md.append("Context mode suppressed (frames): " + ", ".join(f"{k}: {v}" for k, v in suppressed.items()))
    (out / "summary.md").write_text("\n".join(md))
    (out / "summary.json").write_text(json.dumps({"sessions": used, "duration_s": sum(durations),
                                                  "thresholds": args.thresholds, "results": summary}, indent=1))
    print("\n".join(md))
    print(f"-> {out / 'summary.md'}")


if __name__ == "__main__":
    main()
