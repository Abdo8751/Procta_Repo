"""Draw the seat zones of a fixed camera view and save them in configs/exam_context.json.

  python scripts/define_seats.py --video data/videos/Test_1.mp4                # point T1, frame at 5 s
  python scripts/define_seats.py --video data/videos/Test_1.mp4 --auto --t 60   # start from a detected draft

The window shows one frame of the video. Pick a seat with its key, then drag a box over
where that student sits (head to desk edge):
  1 2 3   F1 F2 F3   front row, image left -> right
  4 5 6   B1 B2 B3   back row,  image left -> right
  7       invigilator zone (optional: e.g. the invigilator's desk / standing area)
  x  delete the selected zone      a  auto-draft from person detections (overwrites)
  s  save                          q  quit (unsaved changes are saved)
Zones are stored normalised (0..1), so they work at any resolution of the same view.
--headless --auto saves the automatic draft without a window.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from procta.context import DEFAULT_PATH  # noqa: E402

SEATS = ["F1", "F2", "F3", "B1", "B2", "B3", "invigilator"]
KEYS = {ord(str(i + 1)): s for i, s in enumerate(SEATS)}


def auto_draft(frame, imgsz=1280):
    """Propose 6 seats from YOLOv8-pose person boxes: 3 lowest people = front row, the next
    3 = back row, each sorted left to right. The draft is a starting point to correct."""
    from ultralytics import YOLO
    w = ROOT / "models/yolov8n-pose.pt"
    r = YOLO(str(w) if w.exists() else w.name).predict(frame, classes=[0], imgsz=imgsz, conf=0.25, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    h, W = frame.shape[:2]
    boxes = sorted(boxes.tolist(), key=lambda b: -(b[3]))          # lowest box bottom first = nearest row
    rows = [sorted(boxes[:3], key=lambda b: b[0]), sorted(boxes[3:6], key=lambda b: b[0])]
    zones = {}
    for prefix, row in zip("FB", rows):
        for i, b in enumerate(row):
            pw, ph = b[2] - b[0], b[3] - b[1]
            x1, y1 = max(0, b[0] - 0.08 * pw), max(0, b[1] - 0.05 * ph)
            x2, y2 = min(W, b[2] + 0.08 * pw), min(h, b[3] + 0.05 * ph)
            zones[f"{prefix}{i + 1}"] = [round(x1 / W, 4), round(y1 / h, 4), round(x2 / W, 4), round(y2 / h, 4)]
    return zones


def save(ctx_path, point_id, zones, video):
    raw = json.loads(Path(ctx_path).read_text(encoding="utf-8"))
    pts = raw.setdefault("observation_points", [])
    pt = next((p for p in pts if p["id"] == point_id), None)
    if pt is None:
        pt = {"id": point_id, "name": f"Fixed camera ({Path(video).stem})", "dwell_s": 0, "on_route": False,
              "neighbour_side": "", "allowed_look_sides": [], "allowed_look_reason": "", "ignore_zones": []}
        pts.append(pt)
    pt["seat_zones"] = {k: zones[k] for k in SEATS if k in zones}
    Path(ctx_path).write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(pt['seat_zones'])} zones for point {point_id} -> {ctx_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--t", type=float, default=5.0, help="video second of the frame to show")
    ap.add_argument("--point", default="T1")
    ap.add_argument("--context", default=str(DEFAULT_PATH))
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read {args.video} at {args.t}s")
    h, w = frame.shape[:2]
    raw = json.loads(Path(args.context).read_text(encoding="utf-8"))
    pt = next((p for p in raw.get("observation_points", []) if p["id"] == args.point), {})
    zones = dict(pt.get("seat_zones", {}))
    if args.auto:
        zones = auto_draft(frame)
    if args.headless:
        save(args.context, args.point, zones, args.video)
        return

    scale = min(1.0, 1500 / w, 850 / h)
    view = cv2.resize(frame, None, fx=scale, fy=scale)
    state = {"sel": "F1", "drag": None, "dirty": args.auto}

    def on_mouse(ev, x, y, flags, _):
        nx, ny = x / view.shape[1], y / view.shape[0]
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (nx, ny, nx, ny)
        elif ev == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["drag"] = state["drag"][:2] + (nx, ny)
        elif ev == cv2.EVENT_LBUTTONUP and state["drag"]:
            x1, y1, x2, y2 = state["drag"]
            state["drag"] = None
            if abs(x2 - x1) > 0.01 and abs(y2 - y1) > 0.01:
                zones[state["sel"]] = [round(min(x1, x2), 4), round(min(y1, y2), 4),
                                       round(max(x1, x2), 4), round(max(y1, y2), 4)]
                state["dirty"] = True
                nxt = SEATS.index(state["sel"]) + 1          # move on to the next seat
                if nxt < len(SEATS) - 1:
                    state["sel"] = SEATS[nxt]

    win = "PROCTA seats - 1..6 seat, 7 invigilator, drag a box, x delete, a auto, s save, q quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = view.copy()
        H, W = img.shape[:2]
        for sid, z in zones.items():
            col = (0, 0, 255) if sid == state["sel"] else ((200, 120, 255) if sid == "invigilator" else (0, 220, 0))
            cv2.rectangle(img, (int(z[0] * W), int(z[1] * H)), (int(z[2] * W), int(z[3] * H)), col, 2)
            cv2.putText(img, sid, (int(z[0] * W) + 5, int(z[1] * H) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        if state["drag"]:
            x1, y1, x2, y2 = state["drag"]
            cv2.rectangle(img, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (0, 0, 255), 1)
        cv2.putText(img, f"selected: {state['sel']}  (drag to draw)   zones: {len(zones)}/7"
                         f"{'  *unsaved*' if state['dirty'] else ''}", (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k in KEYS:
            state["sel"] = KEYS[k]
        elif k == ord("x") and state["sel"] in zones:
            del zones[state["sel"]]
            state["dirty"] = True
        elif k == ord("a"):
            zones.clear()
            zones.update(auto_draft(frame))
            state["dirty"] = True
        elif k == ord("s"):
            save(args.context, args.point, zones, args.video)
            state["dirty"] = False
        elif k == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            if state["dirty"]:
                print("Unsaved changes - saving.")
                save(args.context, args.point, zones, args.video)
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
