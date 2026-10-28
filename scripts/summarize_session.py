"""All four methods x both object detectors on one recorded session -> results/<name>/summary.md

  python scripts/summarize_session.py --name test_1 --point T1 \
      --detections world=data/detections/Test_1_world.jsonl coco=data/detections/Test_1_coco.jsonl
  ... --gt data/labels/Test_1_gt.csv      # adds precision / recall / false alerts per hour / delay per event and seat

Without ground truth the summary reports what can be measured honestly: alert counts per
method, event type and seat (the review load an invigilator would face), and what the
context mode logged instead of alerting.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from procta.config import EVENT_TYPES, Config  # noqa: E402
from procta.context import ExamContext  # noqa: E402
from procta.detect import load_detections  # noqa: E402
from procta.evaluation import alert_rows, evaluate, evaluate_per_seat, load_gt, to_markdown  # noqa: E402
from procta.temporal import MODES, run_policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--point", default="T1")
    ap.add_argument("--detections", nargs="+", required=True, help="label=path.jsonl ...")
    ap.add_argument("--gt", default=None)
    args = ap.parse_args()

    ctx = ExamContext.load()
    pt = ctx.point(args.point)
    cfg = Config()
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    gt = load_gt(args.gt) if args.gt else None
    lines = [f"# Session `{args.name}` - all methods, observation point {args.point}\n"]
    data = {}
    for spec in args.detections:
        label, path = spec.split("=", 1)
        meta, frames = load_detections(Path(path))
        dur = frames[-1]["t"]
        lines.append(f"## Object detector: `{label}`  ({dur / 60:.1f} min, {len(frames)} analysed frames, "
                     f"stride {meta.get('stride', 1)}, extraction {meta.get('processing_s', '?')} s)\n")
        lines.append("| method | alerts | per minute | " + " | ".join(EVENT_TYPES) + " |")
        lines.append("|---" * (len(EVENT_TYPES) + 3) + "|")
        per_seat = {}
        for m in MODES:
            alerts, _, sig = run_policy(frames, m, cfg, context=ctx, point=pt, return_events=True)
            c = Counter(a.event_type for a in alerts)
            lines.append(f"| {m} | {len(alerts)} | {len(alerts) / dur * 60:.1f} | "
                         + " | ".join(str(c.get(e, 0)) for e in EVENT_TYPES) + " |")
            per_seat[m] = Counter((str(a.track_id), a.event_type) for a in alerts)
            data[(label, m)] = {"alerts": [vars(a) for a in alerts], "suppressed": dict(sig.suppressed)}
            if gt is not None:
                rows = alert_rows(alerts)
                data[(label, m)]["metrics"] = evaluate(rows, gt, dur, cfg.persistence_s, 1.0, EVENT_TYPES,
                                                       cfg.event_persistence_s)
                data[(label, m)]["per_seat"] = evaluate_per_seat(rows, gt, dur, cfg.persistence_s, 1.0,
                                                                 EVENT_TYPES, cfg.event_persistence_s)
        lines.append("")
        seats = sorted({s for m in MODES for s, _ in per_seat[m]})
        lines.append("Alerts per seat (context mode):\n")
        lines.append("| seat | " + " | ".join(EVENT_TYPES) + " |")
        lines.append("|---" * (len(EVENT_TYPES) + 1) + "|")
        for s in seats:
            lines.append(f"| {s} | " + " | ".join(str(per_seat["context"].get((s, e), 0)) for e in EVENT_TYPES) + " |")
        supp = data[(label, "context")]["suppressed"]
        lines.append(f"\nLogged instead of alerted (context mode, frame counts): {supp or 'none'}\n")
        if gt is not None:
            for m in MODES:
                lines.append(to_markdown(data[(label, m)]["metrics"], f"{label} / {m} - per event type") + "\n")
            lines.append(f"### {label} / context - per seat\n")
            for s, r in data[(label, "context")]["per_seat"].items():
                lines.append(to_markdown({k: v for k, v in r.items() if k != "ALL" and (v["alerts"] or v["target_episodes"])},
                                         f"seat {s}") + "\n")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({f"{k[0]}/{k[1]}": v for k, v in data.items()}, indent=1, default=str),
                                      encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out / 'summary.md'}")


if __name__ == "__main__":
    main()
