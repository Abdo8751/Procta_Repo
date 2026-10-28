"""Keyboard ground-truth labeller for recorded sessions (single participant or a 6-seat group).

Loads an existing GT file (e.g. the one record_session.py or draft_labels.py derived from the
protocol) so you only need to correct it to what actually happened on video.

GT format: event_type,start_s,end_s,seat,note   (seat empty for single-participant sessions;
           two seats joined with "+" for exchanges, e.g. B2+F2; the invigilator is INV)

Keys:
  space       play / pause            a / d   back / forward 1 s     j / l  back / forward 5 s
  label keys  press once to START a label for the selected seat, again to END it:
              1 phone_in_hand   2 phone_on_desk   3 body_movement   4 unusual_seat_movement
              5 head_turned     6 looking_down    7 material_exchange (selected seat + partner)
              8 lookalike_calculator   9 lookalike_clock   0 lookalike_authorised_exchange (seat + partner)
  seat        LEFT click on a person = select seat, RIGHT click = select partner (exchanges)
              or  [ / ]  previous / next seat,  - / =  previous / next partner
  x           delete the label(s) under the current time (for the selected seat, if any)
  s           save                    q       save and quit

Usage:
  python scripts/label_video.py --video data/videos/s01.mp4                 # single participant
  python scripts/label_video.py --video data/videos/Test_1.mp4 --point T1   # group: seats from point T1
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEYS = {ord("1"): "phone_in_hand", ord("2"): "phone_on_desk", ord("3"): "body_movement",
        ord("4"): "unusual_seat_movement", ord("5"): "head_turned", ord("6"): "looking_down",
        ord("7"): "material_exchange", ord("8"): "lookalike_calculator", ord("9"): "lookalike_clock",
        ord("0"): "lookalike_authorised_exchange"}
PAIR_EVENTS = {"material_exchange", "lookalike_authorised_exchange"}
COLORS = {"phone_in_hand": (0, 0, 255), "phone_on_desk": (0, 140, 255),
          "body_movement": (255, 0, 200), "unusual_seat_movement": (255, 200, 0),
          "head_turned": (0, 255, 255), "looking_down": (60, 60, 255), "material_exchange": (0, 200, 0),
          "lookalike_calculator": (160, 160, 160), "lookalike_clock": (200, 200, 200),
          "lookalike_authorised_exchange": (120, 200, 120)}


def seat_label(ev, seat, partner):
    if not seat:
        return ""
    if ev in PAIR_EVENTS and partner and partner != seat:
        return "+".join(sorted([seat, partner]))
    return seat


def load_labels(path):
    labels = []
    if path.exists():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            labels.append([r["event_type"], float(r["start_s"]), float(r["end_s"]),
                           (r.get("seat") or "").strip(), r.get("note", "")])
    return labels


def save_labels(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_type", "start_s", "end_s", "seat", "note"])
        for ev, s, e, seat, note in sorted(labels, key=lambda x: (x[1], x[3])):
            w.writerow([ev, f"{s:.2f}", f"{e:.2f}", seat, note])
    print(f"saved {len(labels)} labels -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--point", default=None, help="observation point with seat zones (group sessions), e.g. T1")
    ap.add_argument("--context", default=str(ROOT / "configs/exam_context.json"))
    args = ap.parse_args()
    video = Path(args.video)
    gt_path = Path(args.gt) if args.gt else ROOT / "data/labels" / f"{video.stem}_gt.csv"

    zones = {}
    if args.point:
        raw = json.loads(Path(args.context).read_text(encoding="utf-8"))
        zones = next((p.get("seat_zones", {}) for p in raw["observation_points"] if p["id"] == args.point), {})
    seats = [s for s in ("F1", "F2", "F3", "B1", "B2", "B3") if s in zones] + ["INV"] if zones else []
    sel = {"seat": seats[0] if seats else "", "partner": seats[1] if len(seats) > 1 else ""}

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tf = video.with_name(video.stem + "_times.csv")
    times = [float(r["t"]) for r in csv.DictReader(open(tf))] if tf.exists() else [i / fps for i in range(n)]
    n = min(n, len(times))
    dur = times[n - 1]

    labels = load_labels(gt_path)
    open_start = {}                       # (event, seat label) -> start time
    view = {"w": 1, "h": 1}

    def on_mouse(ev, x, y, flags, _):
        if ev not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) or not zones or y >= view["h"]:
            return
        fx, fy = x / view["w"], y / view["h"]
        hit = [s for s, z in zones.items() if s != "invigilator" and z[0] <= fx <= z[2] and z[1] <= fy <= z[3]]
        hit = min(hit, key=lambda s: (zones[s][2] - zones[s][0]) * (zones[s][3] - zones[s][1])) if hit else "INV"
        sel["seat" if ev == cv2.EVENT_LBUTTONDOWN else "partner"] = hit

    win = "PROCTA labeller (keys 0-9 label, click seat, space play, a/d j/l seek, x delete, q quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    rows = list(COLORS)
    idx, playing = 0, False
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx = max(0, idx - 1)
            playing = False
            continue
        t = times[idx]
        scale = min(1.0, 1400 / frame.shape[1])
        frame = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        h, w = frame.shape[:2]
        view.update(w=w, h=h)
        for s, z in zones.items():
            if s == "invigilator":
                continue
            col = (0, 0, 255) if s == sel["seat"] else ((0, 200, 0) if s == sel["partner"] else (150, 150, 150))
            cv2.rectangle(frame, (int(z[0] * w), int(z[1] * h)), (int(z[2] * w), int(z[3] * h)), col,
                          1 if col == (150, 150, 150) else 2)
            cv2.putText(frame, s, (int(z[0] * w) + 4, int(z[1] * h) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        bar = np.full((8 + 12 * len(rows), w, 3), 30, np.uint8)
        for row, ev in enumerate(rows):
            y = 4 + row * 12
            for e, s, en, seat, _ in labels:
                if e == ev and (not sel["seat"] or not seat or sel["seat"] in seat.split("+")):
                    cv2.rectangle(bar, (int(s / dur * w), y), (int(en / dur * w), y + 8), COLORS[ev], -1)
            for (oe, _), st in open_start.items():
                if oe == ev:
                    cv2.rectangle(bar, (int(st / dur * w), y), (int(t / dur * w), y + 8), COLORS[ev], 1)
        cv2.line(bar, (int(t / dur * w), 0), (int(t / dur * w), bar.shape[0]), (255, 255, 255), 1)
        active = [f"{e}@{seat}" if seat else e for e, s, en, seat, _ in labels if s <= t <= en]
        cv2.putText(frame, f"{t:6.1f}s / {dur:.0f}s  {'PLAY' if playing else 'PAUSE'}"
                           f"{'  seat ' + sel['seat'] + '  partner ' + sel['partner'] if seats else ''}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "active: " + ", ".join(active)[:150], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        if open_start:
            cv2.putText(frame, "recording: " + ", ".join(f"{e}@{s}" if s else e for e, s in open_start), (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(win, np.vstack([frame, bar]))

        k = cv2.waitKey(int(1000 / fps) if playing else 30) & 0xFF
        if k == ord("q"):
            save_labels(gt_path, labels)
            break
        elif k == ord(" "):
            playing = not playing
        elif k in (ord("a"), ord("d"), ord("j"), ord("l")):
            step = {ord("a"): -1, ord("d"): 1, ord("j"): -5, ord("l"): 5}[k]
            idx = int(np.clip(np.searchsorted(times[:n], t + step), 0, n - 1))
            continue
        elif seats and k in (ord("["), ord("]"), ord("-"), ord("=")):
            key = "seat" if k in (ord("["), ord("]")) else "partner"
            step = 1 if k in (ord("]"), ord("=")) else -1
            sel[key] = seats[(seats.index(sel[key]) + step) % len(seats)] if sel[key] in seats else seats[0]
        elif k in KEYS:
            ev = KEYS[k]
            seat = seat_label(ev, sel["seat"], sel["partner"])
            if (ev, seat) in open_start:
                s = open_start.pop((ev, seat))
                labels.append([ev, min(s, t), max(s, t), seat, "manual"])
                print(f"label {ev} {seat}: {min(s, t):.1f}-{max(s, t):.1f}s")
            else:
                open_start[(ev, seat)] = t
        elif k == ord("x"):
            labels = [lb for lb in labels if not (lb[1] <= t <= lb[2] and
                                                  (not sel["seat"] or not lb[3] or sel["seat"] in lb[3].split("+")))]
        elif k == ord("s"):
            save_labels(gt_path, labels)
        if playing:
            idx = min(idx + 1, n - 1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
