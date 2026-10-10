"""Replay cached detections through the BASELINE and the IMPROVED alerting policy,
evaluate both against ground truth, and build evidence + a review page.

Usage:
  python scripts/run_offline.py --detections data/detections/s01.jsonl \
      --gt data/labels/s01_gt.csv --video data/videos/s01.mp4

Outputs in results/<session>/:
  alerts_baseline.csv, alerts_improved.csv
  metrics.md, metrics.json             (if --gt given)
  evidence/*.jpg + review_improved.html, review_baseline.html   (if --video given)
"""

import argparse
import bisect
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from procta.config import Config, EVENT_TYPES  # noqa: E402
from procta.detect import load_detections  # noqa: E402
from procta.evaluation import evaluate, load_gt, to_markdown  # noqa: E402
from procta.temporal import run_policy  # noqa: E402

FIELDS = ["track_id", "event_type", "t_alert", "t_start", "confidence", "positive_ratio"]


def write_alerts(path, alerts):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for a in alerts:
            r = a.to_row()
            r["t_alert"], r["t_start"] = f"{a.t_alert:.3f}", f"{a.t_start:.3f}"
            w.writerow(r)


def make_evidence(video, frames, alerts, out_dir, tag):
    """Evidence = side-by-side image: frame at episode start | frame when the alert fired."""
    import cv2
    import numpy as np
    from procta.draw import annotate

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = [f["t"] for f in frames]
    cap = cv2.VideoCapture(str(video))
    files = []

    def grab(t):
        k = min(bisect.bisect_left(ts, t), len(frames) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[k]["frame"])
        ok, img = cap.read()
        return (annotate(img, frames[k]) if ok else None), frames[k]["t"]

    for i, a in enumerate(alerts):
        (s_img, s_t), (a_img, a_t) = grab(a.t_start), grab(a.t_alert)
        if s_img is None or a_img is None:
            files.append("")
            continue
        for img, label in ((s_img, f"episode start {s_t:.1f}s"), (a_img, f"alert {a_t:.1f}s")):
            cv2.rectangle(img, (0, img.shape[0] - 30), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
            cv2.putText(img, f"{label} | track {a.track_id} {a.event_type}", (8, img.shape[0] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        pair = np.hstack([s_img, a_img])
        scale = 1400 / pair.shape[1]
        pair = cv2.resize(pair, None, fx=scale, fy=scale) if scale < 1 else pair
        name = f"{tag}_{i:03d}_{a.event_type}_{a.t_alert:.1f}s.jpg"
        cv2.imwrite(str(out_dir / name), pair, [cv2.IMWRITE_JPEG_QUALITY, 80])
        files.append(f"evidence/{name}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tolerance", type=float, default=1.0)
    args = ap.parse_args()

    meta, frames = load_detections(Path(args.detections))
    session = Path(args.detections).stem
    out = Path(args.out) if args.out else ROOT / "results" / session
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    duration = meta.get("duration_s") or frames[-1]["t"]

    report = [f"# Session `{session}`  ({duration / 60:.1f} min, {len(frames)} frames)\n"]
    all_metrics = {}
    for mode in ("baseline", "improved"):
        alerts = run_policy(frames, mode, cfg)
        write_alerts(out / f"alerts_{mode}.csv", alerts)
        print(f"{mode:9s}: {len(alerts)} alerts")
        if args.gt:
            rows = [{"event_type": a.event_type, "t_alert": a.t_alert} for a in alerts]
            m = evaluate(rows, load_gt(args.gt), duration, cfg.persistence_s, args.tolerance, EVENT_TYPES)
            all_metrics[mode] = m
            report.append(to_markdown(m, mode) + "\n")
        if args.video:
            files = make_evidence(args.video, frames, alerts, out / "evidence", mode)
            from procta.review import write_review_page
            write_review_page(out / f"review_{mode}.html", alerts, files,
                              f"PROCTA review - {session} ({mode})", sort_by_confidence=(mode == "improved"))

    report.append("\nConfig:\n```json\n" + json.dumps(cfg.to_dict(), indent=1) + "\n```\n")
    (out / "metrics.md").write_text("\n".join(report))
    if all_metrics:
        (out / "metrics.json").write_text(json.dumps(all_metrics, indent=1))
        print("\n".join(report[1:-1]))
    print(f"Results -> {out}")


if __name__ == "__main__":
    main()
