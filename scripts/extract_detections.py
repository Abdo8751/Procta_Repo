"""Run YOLOv8 (phones) + YOLOv8-pose (people) over a recorded video ONCE and cache the
per-frame detections. All alerting policies are then replayed from this cache.

Usage:
  python scripts/extract_detections.py --video data/videos/s01.mp4
  -> data/detections/s01.jsonl
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from procta.config import Config  # noqa: E402
from procta.detect import Detector, save_detections  # noqa: E402


def frame_times(video: Path, n_frames: int, fps: float):
    tf = video.with_name(video.stem + "_times.csv")
    if tf.exists():
        with open(tf) as f:
            return [float(r["t"]) for r in csv.DictReader(f)]
    return [i / fps for i in range(n_frames)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="for quick tests")
    args = ap.parse_args()

    video = Path(args.video)
    cfg = Config(imgsz=args.imgsz, det_model=str(ROOT / "models/yolov8n.pt"),
                 pose_model=str(ROOT / "models/yolov8n-pose.pt"))
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    times = frame_times(video, n, fps)
    det = Detector(cfg)

    frames, i, t0 = [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(times) or (args.max_frames and i >= args.max_frames):
            break
        fr = det(frame, times[i])
        fr["frame"] = i
        frames.append(fr)
        i += 1
        if i % 100 == 0:
            print(f"  {i}/{n} frames  ({i / (time.time() - t0):.1f} fps processing)")

    out = Path(args.out) if args.out else ROOT / "data/detections" / f"{video.stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"video": str(video), "n_frames": len(frames), "duration_s": frames[-1]["t"] if frames else 0,
            "config": cfg.to_dict()}
    save_detections(out, meta, frames)
    print(f"Saved {len(frames)} frames of detections -> {out}")


if __name__ == "__main__":
    main()
