"""Run YOLOv8 (phones) + YOLOv8-pose (people) over a recorded video ONCE and cache the
per-frame detections. All alerting policies are then replayed from this cache.

Usage:
  python scripts/extract_detections.py --video data/videos/s01.mp4
  -> data/detections/s01.jsonl
  python scripts/extract_detections.py --video data/videos/Test_1.mp4 --stride 3 --imgsz 640 --det-imgsz 1280
  python scripts/extract_detections.py --video data/videos/Test_1.mp4 --start 60 --seconds 30 --stride 3 ...
      (speed test on a 30 s slice; nothing is overwritten unless --out is given)

--stride N processes every Nth frame; skipped frames are only grabbed (not converted) and
every processed frame keeps its real video timestamp, so temporal thresholds stay in seconds.
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
    ap.add_argument("--imgsz", type=int, default=416, help="pose model input size")
    ap.add_argument("--det-imgsz", type=int, default=0, help="object detector input size (0 = --imgsz)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--object-detector", choices=["coco", "world"], default="coco",
                    help="coco = YOLOv8n cell phone; world = YOLO-World + CLIP phone/calculator check")
    ap.add_argument("--det-skip", type=int, default=1, help="run the object detector on every Nth processed frame")
    ap.add_argument("--start", type=float, default=0.0, help="start at this video second")
    ap.add_argument("--seconds", type=float, default=None, help="only process this many seconds")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="for quick tests")
    args = ap.parse_args()

    video = Path(args.video)
    cfg = Config(imgsz=args.imgsz, det_imgsz=args.det_imgsz, det_model=str(ROOT / "models/yolov8n.pt"),
                 pose_model=str(ROOT / "models/yolov8n-pose.pt"), object_detector=args.object_detector,
                 det_frame_skip=args.det_skip, world_model=str(ROOT / "models/yolov8s-worldv2.pt"))
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    times = frame_times(video, n, fps)
    det = Detector(cfg)

    i = int(round(args.start * fps))
    if i:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    end_t = args.start + args.seconds if args.seconds else float("inf")
    todo = (min(n, int(end_t * fps)) - i) // args.stride if args.seconds else (n - i) // args.stride
    frames, t0 = [], time.time()
    while i < len(times) and times[i] < end_t and not (args.max_frames and len(frames) >= args.max_frames):
        if (i - int(round(args.start * fps))) % args.stride:
            if not cap.grab():                 # skipped frame: decode only, no conversion
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        fr = det(frame, times[i])
        fr["frame"] = i
        frames.append(fr)
        i += 1
        if len(frames) % 50 == 0:
            el = time.time() - t0
            rate = len(frames) / el
            print(f"  {len(frames)}/{todo} processed frames  video t={times[i - 1]:.0f}s  "
                  f"{rate:.2f} frames/s  ETA {max(todo - len(frames), 0) / rate / 60:.1f} min", flush=True)

    el = time.time() - t0
    out = Path(args.out) if args.out else ROOT / "data/detections" / f"{video.stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"video": str(video), "n_frames": len(frames), "duration_s": frames[-1]["t"] if frames else 0,
            "stride": args.stride, "start_s": args.start, "fps": fps, "processing_s": round(el, 1),
            "config": cfg.to_dict()}
    print(f"{len(frames)} frames in {el:.0f} s = {el / max(len(frames), 1):.2f} s/frame; "
          f"whole video at this setting ~{el / max(len(frames), 1) * n / args.stride / 60:.0f} min")
    save_detections(out, meta, frames)
    print(f"Saved {len(frames)} frames of detections -> {out}")


if __name__ == "__main__":
    main()
