"""Examination context: the rules of this mock exam and the robot's observation route.

Some events cannot be interpreted without context (supervisor feedback, Oct 2026):
a head turn towards the wall clock is normal, a head turn towards the neighbour's
desk is worth a look; a calculator may be allowed, a phone is not. The context file
(configs/exam_context.json) says which is which. It never turns an event into a
misconduct judgement - it only decides whether an observation is worth a reviewer's time.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "exam_context.json"


@dataclass
class ObservationPoint:
    id: str
    name: str = ""
    dwell_s: float = 120.0
    neighbour_side: str = ""                 # "left" / "right" (image side) or "" if no neighbour visible
    allowed_look_sides: list = field(default_factory=list)
    allowed_look_reason: str = ""
    ignore_zones: list = field(default_factory=list)   # [x1, y1, x2, y2] as fractions of the image


@dataclass
class ExamContext:
    exam: str = ""
    permitted_devices: list = field(default_factory=list)
    permitted_materials: list = field(default_factory=list)
    authorised_interactions: list = field(default_factory=list)
    seats: list = field(default_factory=list)
    insufficient_view_s: float = 4.0
    consolidation_window_s: float = 30.0
    observation_points: list = field(default_factory=list)

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        pts = [ObservationPoint(**{k: v for k, v in p.items() if k in ObservationPoint.__dataclass_fields__})
               for p in raw.get("observation_points", [])]
        kw = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__ and k != "observation_points"}
        return cls(observation_points=pts or [ObservationPoint("A")], **kw)

    def point(self, pid):
        return next((p for p in self.observation_points if p.id == pid), self.observation_points[0])

    def device_permitted(self, coco_name: str) -> bool:
        return coco_name in self.permitted_devices

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)


def in_zone(box, zones, wh):
    """True if the centre of box (pixels) lies in any normalised zone."""
    if not zones or not wh:
        return False
    cx, cy = (box[0] + box[2]) / 2.0 / wh[0], (box[1] + box[3]) / 2.0 / wh[1]
    return any(z[0] <= cx <= z[2] and z[1] <= cy <= z[3] for z in zones)
