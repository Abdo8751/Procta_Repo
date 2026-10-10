"""Live webcam demo (replaces procta_live_baseline.py; --mode baseline behaves like it).

  python scripts/live.py --mode improved            # confidence-aware alerts + evidence snapshots
  python scripts/live.py --mode baseline --record   # also save video+timestamps for later evaluation

Alert times are seconds since the session started, so a --record'ed session can be
evaluated directly with run_offline.py.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from procta.config import Config  # noqa: E402
from procta.detect import Detector  # noqa: E402
from procta.draw import annotate  # noqa: E402
from procta.signals import SignalExtractor  # noqa: E402
from procta.temporal import make_policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mode", choices=["baseline", "improved"], default="improved")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--det-skip", type=int, default=2)
    args = ap.parse_args()

    cfg = Config(imgsz=args.imgsz, det_frame_skip=args.det_skip,
                 det_model=str(ROOT / "models/yolov8n.pt"), pose_model=str(ROOT / "models/yolov8n-pose.pt"))
    name = f"live_{datetime.now():%Y%m%d_%H%M%S}_{args.mode}"
    out = ROOT / "results" / name
    (out / "evidence").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}")
    det, sig, pol = Detector(cfg), SignalExtractor(cfg, args.mode), make_policy(args.mode, cfg)

    writer, times = None, []
    log = open(out / "alerts.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["track_id", "event_type", "t_alert", "t_start", "confidence", "positive_ratio", "evidence"])
    recent = []          # (until_t, text) banners
    start, prev = time.time(), time.time()
    print(f"Mode={args.mode}. Sit still ~2 s for seat calibration. Press 'q' to quit. Output -> {out}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            t = now - start
            fps, prev = 1.0 / max(now - prev, 1e-6), now
            if args.record:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
                writer.write(frame)
                times.append(t)

            dets = det(frame, t)
            signals = sig(dets)
            alerts = pol.step(t, signals, {p["id"] for p in dets["persons"]})
            img = annotate(frame, dets, signals)
            for a in alerts:
                ev_file = out / "evidence" / f"{a.t_alert:07.1f}s_track{a.track_id}_{a.event_type}.jpg"
                cv2.imwrite(str(ev_file), img)
                wr.writerow([a.track_id, a.event_type, f"{a.t_alert:.2f}", f"{a.t_start:.2f}",
                             a.confidence, a.positive_ratio, ev_file.name])
                log.flush()
                conf = "" if a.confidence != a.confidence else f" conf {a.confidence:.2f}"
                recent.append((t + 4, f"ALERT track {a.track_id}: {a.event_type}{conf}"))
                print(f"[{t:7.1f}s] ALERT track {a.track_id} {a.event_type}{conf}")
            recent = [r for r in recent if r[0] > t]
            img = annotate(frame, dets, signals, [r[1] for r in recent])
            cv2.putText(img, f"FPS {fps:.1f}  t={t:.0f}s  mode={args.mode}", (10, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow("PROCTA live", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        log.close()
        if writer is not None:
            writer.release()
            with open(out / f"{name}_times.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "t"])
                w.writerows([i, f"{x:.4f}"] for i, x in enumerate(times))
        cv2.destroyAllWindows()
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
