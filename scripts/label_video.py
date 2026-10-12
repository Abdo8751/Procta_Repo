"""Keyboard ground-truth labeller for recorded sessions.

Loads an existing GT file (e.g. the one record_session.py derived from the protocol)
so you only need to correct it to what actually happened on video.

Keys:
  space       play / pause            a / d   back / forward 1 s     j / l  back / forward 5 s
  1 2 3 4 5   start or end a label at the current time:
              1=phone_in_hand  2=phone_on_desk  3=body_movement  4=unusual_seat_movement
              5=head_turned
  x           delete the label(s) under the current time
  s           save                    q       save and quit

Usage:
  python scripts/label_video.py --video data/videos/s01.mp4
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = {ord("1"): "phone_in_hand", ord("2"): "phone_on_desk", ord("3"): "body_movement",
        ord("4"): "unusual_seat_movement", ord("5"): "head_turned"}
COLORS = {"phone_in_hand": (0, 0, 255), "phone_on_desk": (0, 140, 255),
          "body_movement": (255, 0, 200), "unusual_seat_movement": (255, 200, 0),
          "head_turned": (0, 255, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", default=None)
    args = ap.parse_args()
    video = Path(args.video)
    gt_path = Path(args.gt) if args.gt else ROOT / "data/labels" / f"{video.stem}_gt.csv"

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tf = video.with_name(video.stem + "_times.csv")
    times = [float(r["t"]) for r in csv.DictReader(open(tf))] if tf.exists() else [i / fps for i in range(n)]
    n = min(n, len(times))
    dur = times[n - 1]

    labels = []
    if gt_path.exists():
        for r in csv.DictReader(open(gt_path)):
            labels.append([r["event_type"], float(r["start_s"]), float(r["end_s"]), r.get("note", "")])
    open_start = {}

    def save():
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event_type", "start_s", "end_s", "note"])
            for ev, s, e, note in sorted(labels, key=lambda x: x[1]):
                w.writerow([ev, f"{s:.2f}", f"{e:.2f}", note])
        print(f"saved {len(labels)} labels -> {gt_path}")

    idx, playing = 0, False
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx = max(0, idx - 1)
            playing = False
            continue
        t = times[idx]
        h, w = frame.shape[:2]
        bar = np.full((70, w, 3), 30, np.uint8)
        for row, ev in enumerate(COLORS):
            y = 6 + row * 14
            for e, s, en, _ in labels:
                if e == ev:
                    cv2.rectangle(bar, (int(s / dur * w), y), (int(en / dur * w), y + 10), COLORS[ev], -1)
            if ev in open_start:
                cv2.rectangle(bar, (int(open_start[ev] / dur * w), y), (int(t / dur * w), y + 10), COLORS[ev], 1)
        cv2.line(bar, (int(t / dur * w), 0), (int(t / dur * w), 70), (255, 255, 255), 1)
        active = [e for e, s, en, _ in labels if s <= t <= en]
        cv2.putText(frame, f"{t:6.1f}s / {dur:.0f}s  {'PLAY' if playing else 'PAUSE'}  active: {', '.join(active)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if open_start:
            cv2.putText(frame, "recording label: " + ", ".join(open_start), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("PROCTA labeller (1-4 label, space play, a/d j/l seek, x delete, q quit)", np.vstack([frame, bar]))

        k = cv2.waitKey(int(1000 / fps) if playing else 0) & 0xFF
        if k == ord("q"):
            save()
            break
        elif k == ord(" "):
            playing = not playing
        elif k in (ord("a"), ord("d"), ord("j"), ord("l")):
            step = {ord("a"): -1, ord("d"): 1, ord("j"): -5, ord("l"): 5}[k]
            idx = int(np.clip(np.searchsorted(times[:n], t + step), 0, n - 1))
            continue
        elif k in KEYS:
            ev = KEYS[k]
            if ev in open_start:
                s = open_start.pop(ev)
                labels.append([ev, min(s, t), max(s, t), "manual"])
                print(f"label {ev}: {min(s, t):.1f}-{max(s, t):.1f}s")
            else:
                open_start[ev] = t
        elif k == ord("x"):
            labels = [l for l in labels if not (l[1] <= t <= l[2])]
        elif k == ord("s"):
            save()
        if playing:
            idx = min(idx + 1, n - 1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
