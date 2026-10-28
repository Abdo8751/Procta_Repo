"""Perception layer: phone / object detection + YOLOv8-pose person tracking.

object_detector="coco":  YOLOv8n 'cell phone' class only (original; calculators become phones).
object_detector="world": YOLO-World (open vocabulary) proposes cell phones, calculators, sheets
                         of paper and books; every phone/calculator box is verified by CLIP
                         (procta/verify.py). Output: "phones" (verified only, for the phone
                         events) and "objects" (everything, incl. calculators and paper).

Output of one frame is a plain dict, so a whole video can be cached to JSONL once
and then replayed through any alerting policy (baseline, improved, parameter
sweeps) in seconds, without running the neural networks again. That is what
makes baseline-vs-improvement comparisons fair: both see identical detections.
"""

import json
import threading
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

        self.world = cfg.object_detector == "world"
        if self.world:
            from ultralytics import YOLOWorld
            from .verify import ClipVerifier
            self.det = YOLOWorld(weights(cfg.world_model))
            self.det.set_classes(list(cfg.world_classes))
            self.verifier = ClipVerifier()
        else:
            self.det = YOLO(weights(cfg.det_model))
        self.pose = YOLO(weights(cfg.pose_model))
        self._frame_idx = 0
        self._cached_phones = []
        self._cached_objects = []
        self._lock = threading.Lock()
        self._job = None                      # async: (frame, t) waiting for the worker
        self._wake = threading.Event()
        if cfg.async_objects:
            threading.Thread(target=self._worker, daemon=True).start()

    def _detect_objects(self, frame, t):
        cfg = self.cfg
        if self.world:
            objects = self._world_objects(frame, t)
            phones = [{"box": o["box"], "conf": o["conf"]} for o in objects if o["cls"] == "cell phone"]
        else:
            r = self.det.predict(frame, classes=[PHONE_CLASS_ID], conf=cfg.phone_conf,
                                 imgsz=cfg.det_imgsz or cfg.imgsz, verbose=False)[0]
            phones = []
            if r.boxes is not None:
                for xyxy, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    phones.append({"box": [round(float(v), 1) for v in xyxy], "conf": round(float(conf), 3)})
            objects = [{"cls": "cell phone", **p} for p in phones]
        return phones, objects

    def _worker(self):
        """Async mode: always works on the newest submitted frame; older ones are skipped."""
        while True:
            self._wake.wait()
            with self._lock:
                job, self._job = self._job, None
                self._wake.clear()
            if job is None:
                continue
            phones, objects = self._detect_objects(*job)
            with self._lock:
                self._cached_phones, self._cached_objects = phones, objects

    def _world_objects(self, frame, t):
        cfg = self.cfg
        r = self.det.predict(frame, conf=cfg.world_conf, imgsz=cfg.det_imgsz or cfg.imgsz, verbose=False)[0]
        cands = []                            # merge boxes that got several class names
        if r.boxes is not None:
            for xyxy, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                box, name, s = [round(float(v), 1) for v in xyxy], r.names[int(c)], float(s)
                same = next((o for o in cands if _iou(o["box"], box) > 0.6), None)
                if same is None:
                    cands.append({"box": box, "det": {name: round(s, 3)}})
                else:
                    same["det"][name] = max(same["det"].get(name, 0.0), round(s, 3))
        objects, verify = [], []
        for o in cands:
            if "cell phone" in o["det"] or "calculator" in o["det"]:
                verify.append(o)
            else:
                name = max(o["det"], key=o["det"].get)
                objects.append({"cls": "paper" if name == "sheet of paper" else name, "box": o["box"],
                                "conf": o["det"][name], "det": o["det"]})
        from .verify import decide
        for o, (_, probs) in zip(verify, self.verifier.classify(frame, [o["box"] for o in verify], t)):
            d = decide(probs, cfg.clip_phone_min)
            if d is None:
                continue                      # pencil case, pen, hand: not an exam-relevant object
            cls, conf = d
            objects.append({"cls": cls, "box": o["box"], "conf": round(conf, 3), "det": o["det"], "clip": probs})
        return objects

    def __call__(self, frame, t: float) -> dict:
        cfg = self.cfg
        if self._frame_idx % cfg.det_frame_skip == 0:
            if cfg.async_objects:
                with self._lock:
                    self._job = (frame.copy(), t)
                    self._wake.set()
            else:
                self._cached_phones, self._cached_objects = self._detect_objects(frame, t)
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
        with self._lock:
            phones, objects = list(self._cached_phones), list(self._cached_objects)
        return {"t": round(t, 4), "wh": [w, h], "persons": persons, "phones": phones, "objects": objects}


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


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
