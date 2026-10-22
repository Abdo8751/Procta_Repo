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

head_turned (both modes): the nose shifts sideways relative to the eyes when the head
turns, so yaw = (nose_x - eye-midpoint x) / eye distance (~0 facing the camera).
Measured against each student's own yaw during seat calibration (camera may be off
to one side). In profile one eye disappears, which also counts as turned. Looking
down is deliberately not an event: it is normal when writing on paper.
"""

from collections import Counter, defaultdict

import numpy as np

from .context import in_zone
from .config import (Config, LEFT_EYE, LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_EYE, RIGHT_SHOULDER,
                     RIGHT_WRIST)


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
    def __init__(self, cfg: Config, mode: str = "improved", context=None, point=None):
        """mode: "baseline" (original rules), "improved" (owner assignment + keypoint gating),
        "context" (= improved + examination context: permitted devices, ignore zones,
        head turns only towards the neighbour's desk). context: procta.context.ExamContext."""
        assert mode in ("baseline", "improved", "context")
        if mode == "context" and context is None:
            from .context import ExamContext
            context = ExamContext.load()
        self.cfg, self.mode, self.context = cfg, mode, context
        self.point = point if point is not None else (context.observation_points[0] if context else None)
        self.suppressed = Counter()             # context mode: observations dropped, by reason
        self.prev_shoulder = {}                 # track -> (xy, t)
        self.seat_samples = defaultdict(list)
        self.seat_baseline = {}
        self.head_samples = defaultdict(list)
        self.head_baseline = {}
        self.head_yaw = {}                      # track -> latest yaw offset (for the live overlay)

    def set_point(self, point):
        """New observation point = new viewpoint: re-calibrate seat and head pose."""
        self.point = point
        self.prev_shoulder.clear()
        self.seat_samples.clear()
        self.seat_baseline.clear()
        self.head_samples.clear()
        self.head_baseline.clear()
        self.head_yaw.clear()

    def _kp(self, person, idx):
        x, y, c = person["kps"][idx]
        xy = np.array([x, y])
        if self.mode == "baseline":
            ok = not np.allclose(xy, 0)
        else:
            ok = c >= self.cfg.keypoint_min_conf and not np.allclose(xy, 0)
        return xy, ok, c

    def _yaw(self, pe):
        """(yaw, confidence) or None if the face is not visible."""
        n, n_ok, nc = self._kp(pe, NOSE)
        le, le_ok, lec = self._kp(pe, LEFT_EYE)
        re, re_ok, rec = self._kp(pe, RIGHT_EYE)
        if not n_ok:
            return None
        if le_ok and re_ok:
            d = np.linalg.norm(le - re)
            if d < 1.0:
                return None
            return (n[0] - (le[0] + re[0]) / 2.0) / d, float(min(nc, lec, rec))
        if le_ok or re_ok:                      # profile: one eye hidden behind the nose
            e, ec = (le, lec) if le_ok else (re, rec)
            return float(np.sign(n[0] - e[0]) or 1.0), float(min(nc, ec))
        return None

    def __call__(self, frame: dict) -> dict:
        cfg, t = self.cfg, frame["t"]
        persons, phones = frame["persons"], frame["phones"]
        out = {}

        ctx = self.context if self.mode == "context" else None
        if ctx is not None:
            kept = []
            for ph in phones:
                if ctx.device_permitted("cell phone"):
                    self.suppressed["phone permitted by exam rules"] += 1
                elif self.point is not None and in_zone(ph["box"], self.point.ignore_zones, frame.get("wh")):
                    self.suppressed["phone in ignore zone (e.g. invigilator desk)"] += 1
                else:
                    kept.append(ph)
            phones = kept

        owners = (assign_phone_owners(phones, persons)
                  if self.mode in ("improved", "context") and cfg.assign_phone_to_owner else None)

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

            # ---- event 5: head turned left/right (vs calibrated yaw)
            yaw = self._yaw(pe)
            if tid not in self.head_baseline:
                if yaw is not None:
                    self.head_samples[tid].append(yaw[0])
                if tid in self.seat_baseline:   # calibrate over the same frames as the seat
                    s = self.head_samples.pop(tid, [])
                    self.head_baseline[tid] = float(np.median(s)) if s else 0.0
            elif yaw is None:
                self.head_yaw.pop(tid, None)         # face not visible: no head signal
            else:
                off = yaw[0] - self.head_baseline[tid]
                self.head_yaw[tid] = off
                if abs(off) > cfg.head_yaw_thr:
                    side = "right" if off > 0 else "left"          # image side the nose moved to
                    reason = None
                    if ctx is not None and self.point is not None:
                        if side in self.point.allowed_look_sides:
                            reason = f"head turn to allowed side ({side})"
                        elif self.point.neighbour_side and side != self.point.neighbour_side:
                            reason = f"head turn away from neighbour ({side})"
                    if reason:
                        self.suppressed[reason] += 1
                    else:
                        out[(tid, "head_turned")] = yaw[1]
        return out
