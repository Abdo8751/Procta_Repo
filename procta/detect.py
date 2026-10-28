"""Perception layer: YOLOv8 phone detection + YOLOv8-pose person tracking.

Output of one frame is a plain dict, so a whole video can be cached to JSONL once
and then replayed through any alerting policy (baseline, improved, parameter
sweeps) in seconds, without running the neural networks again. That is what
makes baseline-vs-improvement comparisons fair: both see identical detections.
"""

import json
from pathlib import Path

import numpy as np

from .config import Config, PERSON_CLASS_ID, PHONE_CLASS_ID


class Detector:
    def __init__(self, cfg: Config):
        from ultralytics import YOLO  # imported lazily so evaluation code runs without torch

        self.cfg = cfg
        # fall back to the bare name -> Ultralytics downloads the weights automatically
        def weights(p):
            return p if Path(p).exists() else Path(p).name

        self.det = YOLO(weights(cfg.det_model))
        self.pose = YOLO(weights(cfg.pose_model))
        self._frame_idx = 0
        self._cached_phones = []

    def __call__(self, frame, t: float) -> dict:
        cfg = self.cfg
        if self._frame_idx % cfg.det_frame_skip == 0:
            r = self.det.predict(frame, classes=[PHONE_CLASS_ID], conf=cfg.phone_conf,
                                 imgsz=cfg.det_imgsz or cfg.imgsz, verbose=False)[0]
            phones = []
            if r.boxes is not None:
                for xyxy, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    phones.append({"box": [round(float(v), 1) for v in xyxy], "conf": round(float(conf), 3)})
            self._cached_phones = phones
        self._frame_idx += 1

        p = self.pose.track(frame, persist=True, classes=[PERSON_CLASS_ID], imgsz=cfg.imgsz,
                            verbose=False)[0]
        persons = []
        if p.boxes is not None and p.boxes.id is not None and p.keypoints is not None:
            ids = p.boxes.id.cpu().numpy().astype(int)
            boxes = p.boxes.xyxy.cpu().numpy()
            confs = p.boxes.conf.cpu().numpy()
            kxy = p.keypoints.xy.cpu().numpy()
            kconf = (p.keypoints.conf.cpu().numpy() if p.keypoints.conf is not None
                     else np.ones(kxy.shape[:2]))
            for i, tid in enumerate(ids):
                persons.append({
                    "id": int(tid),
                    "box": [round(float(v), 1) for v in boxes[i]],
                    "conf": round(float(confs[i]), 3),
                    "kps": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                            for (x, y), c in zip(kxy[i], kconf[i])],
                })
        h, w = frame.shape[:2]
        return {"t": round(t, 4), "wh": [w, h], "persons": persons, "phones": list(self._cached_phones)}


def save_detections(path: Path, meta: dict, frames: list):
    with open(path, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for fr in frames:
            f.write(json.dumps(fr) + "\n")


def load_detections(path: Path):
    with open(path) as f:
        meta = json.loads(f.readline())["meta"]
        frames = [json.loads(line) for line in f if line.strip()]
    return meta, frames
