"""Seat assignment: with several students in view the pose tracker switches ids (occlusion,
people leaning into each other), so identity is taken from WHERE a person sits instead.

Seat zones are normalised boxes [x1, y1, x2, y2] per observation point
(configs/exam_context.json -> observation_points[].seat_zones), drawn with
scripts/define_seats.py. A person gets the seat whose zone contains the centre of their
box; if several do, the zone it overlaps most. Each seat holds at most one person per
frame (best overlap wins).

The invigilator is not an examinee: a person whose centre lies in the optional
"invigilator" zone, or who is clearly standing (box much taller than the seat zone they
are in front of) or outside every seat, is labelled INVIGILATOR and gets no examinee events.
"""

import numpy as np

INVIGILATOR = "INV"


def _norm_box(box, wh):
    return [box[0] / wh[0], box[1] / wh[1], box[2] / wh[0], box[3] / wh[1]]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _contains(z, x, y):
    return z[0] <= x <= z[2] and z[1] <= y <= z[3]


def assign_seats(persons, zones: dict, wh, standing_ratio=1.8):
    """-> one label per person: a seat id, INVIGILATOR, or None (no zones / unassignable).

    zones: {"F1": [x1,y1,x2,y2], ..., "invigilator": [...] (optional)}, normalised.
    standing_ratio: person box height / seat zone height above which the person is
    treated as standing (an invigilator walking between the rows)."""
    if not zones or not wh:
        return [None] * len(persons)
    seat_zones = {k: v for k, v in zones.items() if k.lower() != "invigilator"}
    inv_zone = next((v for k, v in zones.items() if k.lower() == "invigilator"), None)
    labels, claims = [None] * len(persons), {}
    for j, pe in enumerate(persons):
        b = _norm_box(pe["box"], wh)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if inv_zone and _contains(inv_zone, cx, cy):
            labels[j] = INVIGILATOR
            continue
        cands = [(s, _iou(b, z)) for s, z in seat_zones.items() if _contains(z, cx, cy)]
        if not cands:
            labels[j] = INVIGILATOR if seat_zones else None   # walking between the rows
            continue
        seat, iou = max(cands, key=lambda c: c[1])
        z = seat_zones[seat]
        if (b[3] - b[1]) > standing_ratio * (z[3] - z[1]):
            labels[j] = INVIGILATOR
            continue
        claims.setdefault(seat, []).append((iou, j))
    for seat, cl in claims.items():
        cl.sort(reverse=True)
        labels[cl[0][1]] = seat
        for _, j in cl[1:]:                     # second person in one seat: someone standing beside it
            labels[j] = INVIGILATOR
    return labels


def seat_centre_px(zone, wh):
    return np.array([(zone[0] + zone[2]) / 2 * wh[0], (zone[1] + zone[3]) / 2 * wh[1]])
