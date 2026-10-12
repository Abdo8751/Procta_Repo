"""Record a staged, consented mock-exam session from the webcam, following a timed
protocol so that ground truth is known by design.

Writes:
  data/videos/<name>.mp4         raw video (no overlays)
  data/videos/<name>_times.csv   true capture time of every frame (webcam fps is irregular)
  data/labels/<name>_gt.csv      ground truth derived from the protocol
                                 (check/adjust it afterwards with scripts/label_video.py)

The preview window shows the current instruction and a countdown; nothing is drawn
on the recorded frames.

Usage:
  python scripts/record_session.py --name s01_me_daylight
  python scripts/record_session.py --name s02 --protocol docs/protocol_v1.csv --camera 1
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def load_protocol(path):
    with open(path) as f:
        return [{**r, "start_s": float(r["start_s"]), "end_s": float(r["end_s"])} for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--protocol", default=str(ROOT / "docs/protocol_v1.csv"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    steps = load_protocol(args.protocol)
    total = steps[-1]["end_s"]
    vid_dir, lab_dir = ROOT / "data/videos", ROOT / "data/labels"
    vid_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    from procta.camera import open_camera
    cap = open_camera(args.camera, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read from camera")
    h, w = frame.shape[:2]
    fps_guess = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(str(vid_dir / f"{args.name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_guess, (w, h))

    # 5-second countdown so you can get into position
    t0 = time.time()
    while time.time() - t0 < 5:
        ok, frame = cap.read()
        prev = frame.copy()
        cv2.putText(prev, f"Recording starts in {5 - int(time.time() - t0)}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow("PROCTA recorder", prev)
        cv2.waitKey(1)

    times = []
    start = time.time()
    print(f"Recording {total:.0f} s ... press 'q' to abort")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time() - start
        if t > total:
            break
        writer.write(frame)
        times.append(t)
        step = next((s for s in steps if s["start_s"] <= t < s["end_s"]), steps[-1])
        nxt = next((s for s in steps if s["start_s"] > t), None)
        prev = frame.copy()
        color = (0, 0, 255) if step["event_type"] != "none" else (0, 160, 0)
        cv2.putText(prev, f"{t:5.1f}s  NOW: {step['instruction']}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(prev, f"until {step['end_s']:.0f}s", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if nxt and nxt["start_s"] - t < 4:
            cv2.putText(prev, f"NEXT in {nxt['start_s'] - t:.0f}s: {nxt['instruction']}", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.imshow("PROCTA recorder", prev)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Aborted early - ground truth is truncated to the recorded length.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with open(vid_dir / f"{args.name}_times.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "t"])
        wr.writerows([i, f"{t:.4f}"] for i, t in enumerate(times))
    end_t = times[-1] if times else 0
    with open(lab_dir / f"{args.name}_gt.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["event_type", "start_s", "end_s", "note"])
        for s in steps:
            if s["event_type"] != "none" and s["start_s"] < end_t:
                for ev in s["event_type"].split("+"):     # e.g. body_movement+head_turned
                    wr.writerow([ev, s["start_s"], min(s["end_s"], end_t), s["instruction"]])
    fps_real = len(times) / end_t if end_t else 0
    print(f"Saved {len(times)} frames ({end_t:.1f} s, real {fps_real:.1f} fps) -> {vid_dir / args.name}.mp4")
    print(f"Ground truth from protocol -> {lab_dir / args.name}_gt.csv  (review it with label_video.py)")


if __name__ == "__main__":
    main()
