"""Per-frame event signals: turn one frame's detections into
{(identity, event_type): confidence} for every event that is "on" in that frame.
identity = the person's seat (F1, B2, ...) when the observation point has seat zones,
otherwise the tracker id. Seats survive tracker id switches; the invigilator (INV) gets
no examinee events.

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
to one side). In profile one eye disappears, which also counts as turned.

looking_down: writing on one's own paper is normal, so the reference is each student's own
calibrated WRITING pose. Head pitch = (eye height - ear height) / shoulder width: when the
head tips forward the ears rise relative to the eyes (they sit ~8 cm behind them), which is
a larger and steadier change than nose vs eyes. The event needs BOTH the head lowered
beyond the writing pose AND the hands off the desk (wrists not visible = hidden by the desk
edge, or clearly lower than in the writing pose) - i.e. looking at the lap / under the desk.

material_exchange: a wrist of one person and a wrist of a DIFFERENT person (different seats,
also front <-> back row) come within exchange_wrist_k x their mean shoulder width, at a point
between the two people. A "paper" object at the wrists raises the confidence. Identity is
the pair ("F1+F2"). With the invigilator (INV) it is an authorised exchange: in context mode
it is logged as a note, not signalled. It is confirmed after 1 s (config.event_persistence_s).

Calculators (object_detector="world") never produce phone events; in context mode a
calculator at a student's hand is logged as a note when it is a permitted device.
"""

from collections import Counter, defaultdict

import numpy as np

from .context import in_zone
from .seats import INVIGILATOR, assign_seats
from .config import (Config, LEFT_EAR, LEFT_EYE, LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_EAR, RIGHT_EYE,
                     RIGHT_SHOULDER, RIGHT_WRIST)


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
        self.head_yaw = {}                      # identity -> latest yaw offset (for the live overlay)
        self.present = set()                    # identities of the examinees in the last frame
        self.pose_samples = defaultdict(list)   # identity -> [(pitch, wrist height, box width)] during calibration
        self.pose_baseline = {}                 # identity -> (pitch, wrist height) of the writing pose
        self.head_pitch = {}                    # identity -> latest pitch offset (for the live overlay)
        self.seat_width = {}                    # identity -> calibrated person box width (px)
        self.notes = []                         # context mode: (t, identity, kind, text) worth logging, not alerting

    def set_point(self, point):
        """New observation point = new viewpoint: re-calibrate seat and head pose."""
        self.point = point
        self.prev_shoulder.clear()
        self.seat_samples.clear()
        self.seat_baseline.clear()
        self.head_samples.clear()
        self.head_baseline.clear()
        self.head_yaw.clear()
        self.pose_samples.clear()
        self.pose_baseline.clear()
        self.head_pitch.clear()
        self.seat_width.clear()

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

    def _shoulders(self, pe):
        ls, ls_ok, _ = self._kp(pe, LEFT_SHOULDER)
        rs, rs_ok, _ = self._kp(pe, RIGHT_SHOULDER)
        if not (ls_ok and rs_ok):
            return None, None
        sw = float(np.linalg.norm(ls - rs))
        return (ls + rs) / 2.0, (sw if sw > 1.0 else None)

    def _pitch(self, pe, sw):
        """((eye height - ear height) / shoulder width, confidence); grows when the head tips forward."""
        eyes = [self._kp(pe, i) for i in (LEFT_EYE, RIGHT_EYE)]
        ears = [self._kp(pe, i) for i in (LEFT_EAR, RIGHT_EAR)]
        eyes, ears = [e for e in eyes if e[1]], [e for e in ears if e[1]]
        if not eyes or not ears or not sw:
            return None
        ey = np.mean([e[0][1] for e in eyes])
        ay = np.mean([e[0][1] for e in ears])
        return float((ey - ay) / sw), float(min(e[2] for e in eyes + ears))

    def _wrist_height(self, pe, sc, sw):
        """Mean visible wrist height below the shoulders / shoulder width, or None if no wrist visible."""
        ws = [self._kp(pe, i) for i in (LEFT_WRIST, RIGHT_WRIST)]
        ws = [w for w in ws if w[1]]
        if not ws or sc is None or not sw:
            return None
        return float((np.mean([w[0][1] for w in ws]) - sc[1]) / sw)

    def _exchanges(self, persons, idents, objects, t):
        """Pairs of different people whose wrists meet between them -> {(pair, event): confidence}."""
        cfg, out = self.cfg, {}
        info = []
        for pe, ident in zip(persons, idents):
            sc, sw = self._shoulders(pe)
            ws = [self._kp(pe, i) for i in (LEFT_WRIST, RIGHT_WRIST)]
            info.append((ident, sc, sw, [(w[0], w[2]) for w in ws if w[1]]))
        papers = [box_center(o["box"]) for o in objects if o.get("cls") == "paper"]
        for a in range(len(info)):
            for b in range(a + 1, len(info)):
                ia, ca, swa, wa = info[a]
                ib, cb, swb, wb = info[b]
                if ia == ib or ca is None or cb is None or not swa or not swb:
                    continue
                if ia == INVIGILATOR and ib == INVIGILATOR:
                    continue
                pair = "+".join(sorted(str(i) for i in (ia, ib)))
                self.present.add(pair)
                msw = (swa + swb) / 2.0
                seg = cb - ca
                seg_len2 = float(seg @ seg) or 1.0
                best = None
                for pa, qa in wa:
                    for pb, qb in wb:
                        if np.linalg.norm(pa - pb) >= cfg.exchange_wrist_k * msw:
                            continue
                        m = (pa + pb) / 2.0
                        u = float((m - ca) @ seg) / seg_len2          # 0 = at a's shoulders, 1 = at b's
                        perp = np.linalg.norm(m - (ca + u * seg))
                        if 0.15 <= u <= 0.85 and perp <= 1.5 * msw:  # the meeting point is BETWEEN them
                            conf = float(min(qa, qb))
                            if any(np.linalg.norm(p - m) < msw for p in papers):
                                conf = min(1.0, conf + cfg.exchange_paper_bonus)
                            best = max(best or 0.0, conf)
                if best is None:
                    continue
                if INVIGILATOR in (ia, ib) and self.mode == "context":
                    self.suppressed["authorised exchange with invigilator"] += 1
                    self.notes.append((t, pair, "authorised_exchange", "material passed with the invigilator"))
                else:
                    out[(pair, "material_exchange")] = best
        return out

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

        zones = self.point.seat_zones if self.point is not None else {}
        seats = assign_seats(persons, zones, frame.get("wh"), cfg.standing_ratio)
        self.present = set()
        objects = frame.get("objects", [])
        idents = [s or pe["id"] for s, pe in zip(seats, persons)]
        out.update(self._exchanges(persons, idents, objects, t))
        calcs = [o for o in objects if o.get("cls") == "calculator"]
        calc_owner = assign_phone_owners(calcs, persons) if calcs else []
        for j, pe in enumerate(persons):
            pe["seat"] = seats[j]
            tid = pe["key"] = idents[j]
            if tid == INVIGILATOR:
                continue
            self.present.add(tid)
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

            # calculators are never phones; in context mode a permitted one is only logged
            if ctx is not None:
                for k, cb in enumerate(calcs):
                    if calc_owner[k] != j:
                        continue
                    pc, thr = box_center(cb["box"]), box_diag(cb["box"]) * cfg.phone_proximity_factor
                    if (lw_ok and np.linalg.norm(lw - pc) < thr) or (rw_ok and np.linalg.norm(rw - pc) < thr):
                        if ctx.device_permitted("calculator"):
                            self.suppressed["calculator in hand (permitted device)"] += 1
                            self.notes.append((t, tid, "permitted_device", "calculator in hand (permitted)"))

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
            sc, sw = self._shoulders(pe)
            pitch = self._pitch(pe, sw)
            wh_rel = self._wrist_height(pe, sc, sw)
            if tid not in self.seat_baseline:
                self.seat_samples[tid].append(c)
                self.pose_samples[tid].append((pitch[0] if pitch else None, wh_rel, pe["box"][2] - pe["box"][0]))
                if len(self.seat_samples[tid]) >= cfg.calibration_frames:
                    self.seat_baseline[tid] = np.mean(self.seat_samples[tid], axis=0)
                    smp = self.pose_samples.pop(tid, [])
                    ps = [s[0] for s in smp if s[0] is not None]
                    hs = [s[1] for s in smp if s[1] is not None]
                    self.pose_baseline[tid] = (float(np.median(ps)) if ps else None,
                                               float(np.median(hs)) if hs else None)
                    self.seat_width[tid] = float(np.median([s[2] for s in smp]))
            else:
                drift_thr = (cfg.seat_drift_rel * self.seat_width[tid] if pe["seat"] and tid in self.seat_width
                             else cfg.seat_drift_px)
                if np.linalg.norm(c - self.seat_baseline[tid]) > drift_thr:
                    out[(tid, "unusual_seat_movement")] = pe["conf"]

                # ---- event 6: looking down below the desk (beyond the writing pose, hands off the desk)
                p0, h0 = self.pose_baseline.get(tid, (None, None))
                if pitch is None or p0 is None:
                    self.head_pitch.pop(tid, None)
                else:
                    dp = pitch[0] - p0
                    self.head_pitch[tid] = dp
                    hands_off_desk = wh_rel is None or (h0 is not None and wh_rel - h0 > cfg.hands_drop_thr)
                    if dp > cfg.head_pitch_thr and hands_off_desk:
                        out[(tid, "looking_down")] = pitch[1]

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
