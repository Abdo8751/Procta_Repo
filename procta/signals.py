"""Per-frame event signals: turn one frame's detections into
{(track_id, event_type): confidence} for every event that is "on" in that frame.

mode="baseline" reproduces procta_live_baseline.py exactly:
  * a wrist counts if its coordinates are non-zero
  * every phone that is not near a person's wrist counts as phone_on_desk FOR THAT
    PERSON - so with two people in view, one phone in A's hand also produces
    phone_on_desk for B (seen in the live log: track 2 in-hand + track 29 on-desk
    at the same second).

mode="improved" fixes the attribution:
  * each phone is assigned to exactly one owner (the nearest person box);
    phones far from everyone (e.g. on the invigilator's desk) are ignored
  * wrists/shoulders are only used if the pose model's keypoint confidence is high
"""

from collections import defaultdict

import numpy as np

from .config import (Config, LEFT_SHOULDER, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_WRIST)


def box_center(b):
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def box_diag(b):
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def point_to_box_dist(p, b):
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return float(np.hypot(dx, dy))


def assign_phone_owners(phones, persons, max_dist_factor=0.5):
    """Return owner index (into persons) per phone, or -1 if no plausible owner."""
    owners = []
    for ph in phones:
        c = box_center(ph["box"])
        best, best_key = -1, None
        for j, pe in enumerate(persons):
            d_box = point_to_box_dist(c, pe["box"])
            width = pe["box"][2] - pe["box"][0]
            if d_box > max_dist_factor * width:
                continue
            key = (d_box, float(np.linalg.norm(c - box_center(pe["box"]))))
            if best_key is None or key < best_key:
                best, best_key = j, key
        owners.append(best)
    return owners


class SignalExtractor:
    def __init__(self, cfg: Config, mode: str = "improved"):
        assert mode in ("baseline", "improved")
        self.cfg, self.mode = cfg, mode
        self.prev_shoulder = {}                 # track -> (xy, t)
        self.seat_samples = defaultdict(list)
        self.seat_baseline = {}

    def _kp(self, person, idx):
        x, y, c = person["kps"][idx]
        xy = np.array([x, y])
        if self.mode == "baseline":
            ok = not np.allclose(xy, 0)
        else:
            ok = c >= self.cfg.keypoint_min_conf and not np.allclose(xy, 0)
        return xy, ok, c

    def __call__(self, frame: dict) -> dict:
        cfg, t = self.cfg, frame["t"]
        persons, phones = frame["persons"], frame["phones"]
        out = {}

        owners = (assign_phone_owners(phones, persons)
                  if self.mode == "improved" and cfg.assign_phone_to_owner else None)

        for j, pe in enumerate(persons):
            tid = pe["id"]
            lw, lw_ok, _ = self._kp(pe, LEFT_WRIST)
            rw, rw_ok, _ = self._kp(pe, RIGHT_WRIST)

            # ---- events 1 & 2: phone in hand / on desk
            for k, ph in enumerate(phones):
                if owners is not None and owners[k] != j:
                    continue
                pc, thr = box_center(ph["box"]), box_diag(ph["box"]) * cfg.phone_proximity_factor
                near = ((lw_ok and np.linalg.norm(lw - pc) < thr) or
                        (rw_ok and np.linalg.norm(rw - pc) < thr))
                ev = "phone_in_hand" if near else "phone_on_desk"
                out[(tid, ev)] = max(out.get((tid, ev), 0.0), ph["conf"])

            # ---- event 3: body movement (speed, px/s)
            ls, ls_ok, lsc = self._kp(pe, LEFT_SHOULDER)
            rs, rs_ok, rsc = self._kp(pe, RIGHT_SHOULDER)
            if ls_ok and rs_ok:
                sc = (ls + rs) / 2.0
                if tid in self.prev_shoulder:
                    pxy, pt = self.prev_shoulder[tid]
                    speed = np.linalg.norm(sc - pxy) / max(t - pt, 1e-3)
                    if speed > cfg.body_movement_speed:
                        out[(tid, "body_movement")] = float(min(lsc, rsc))
                self.prev_shoulder[tid] = (sc, t)

            # ---- event 4: unusual seat movement (drift from calibrated seat)
            c = box_center(pe["box"])
            if tid not in self.seat_baseline:
                self.seat_samples[tid].append(c)
                if len(self.seat_samples[tid]) >= cfg.calibration_frames:
                    self.seat_baseline[tid] = np.mean(self.seat_samples[tid], axis=0)
            elif np.linalg.norm(c - self.seat_baseline[tid]) > cfg.seat_drift_px:
                out[(tid, "unusual_seat_movement")] = pe["conf"]
        return out
