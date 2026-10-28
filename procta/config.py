"""Central configuration. Every threshold used by the system lives here so that
experiments can state exactly which values produced which numbers."""

from dataclasses import dataclass, field, asdict

# COCO / COCO-keypoint indices
PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67          # 'cell phone'
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
NOSE, LEFT_EYE, RIGHT_EYE = 0, 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4

EVENT_TYPES = ("phone_in_hand", "phone_on_desk", "body_movement", "unusual_seat_movement",
               "head_turned", "looking_down", "material_exchange")


@dataclass
class Config:
    # --- models / inference ---
    det_model: str = "models/yolov8n.pt"
    pose_model: str = "models/yolov8n-pose.pt"
    imgsz: int = 416
    det_imgsz: int = 0                # object-detector input size; 0 = same as imgsz (far back rows need 960+)
    phone_conf: float = 0.25          # minimum detector confidence for a phone box
    det_frame_skip: int = 1           # 1 = run phone detector on every frame (offline default)
    object_detector: str = "coco"     # "coco": YOLOv8n cell-phone class (calculators look like phones)
                                      # "world": YOLO-World proposes phone/calculator/paper/book, CLIP verifies
    world_model: str = "models/yolov8s-worldv2.pt"
    world_classes: tuple = ("cell phone", "calculator", "sheet of paper", "book")
    world_conf: float = 0.1           # low on purpose: YOLO-World proposes, CLIP decides
    clip_phone_min: float = 0.35      # CLIP p(smartphone) needed to keep a box as a phone
    async_objects: bool = False       # live only: object detector in a background thread, so the slow
                                      # world+CLIP step never blocks pose tracking / the display

    # --- per-frame event signals ---
    phone_proximity_factor: float = 1.5     # wrist-to-phone distance < factor * phone diagonal
    keypoint_min_conf: float = 0.3          # ignore wrists/shoulders the pose model is unsure about
    body_movement_speed: float = 150.0      # shoulder-centroid speed, px/s
    seat_drift_px: float = 80.0             # distance from calibrated seat centroid, px (no seat zones)
    seat_drift_rel: float = 0.35            # with seat zones: drift > this x the person's calibrated box width
                                            # (back-row students are half as big in the image as the front row)
    calibration_frames: int = 30
    head_yaw_thr: float = 0.25              # |yaw - calibrated yaw|, yaw = (nose_x - eye-mid x) / eye distance
    standing_ratio: float = 1.8             # person box taller than this x seat zone height = standing (invigilator)
    # looking_down: hands go below the desk while the head stays at (or below) writing level
    head_up_tol: float = 0.05               # pitch - writing pitch must be >= -this (head not raised);
                                            # pitch = (eye y - ear y) / shoulder width
    hands_drop_thr: float = 0.20            # wrists lower than the writing position by this x shoulder width
                                            # (or not visible at all = below the desk edge)
    hands_ref_s: float = 60.0               # writing position = rolling median wrist height of the last 60 s
    # material_exchange: wrists of two different people meet between them
    exchange_wrist_k: float = 0.6           # wrist-to-wrist distance < k x mean shoulder width of the two
    exchange_paper_bonus: float = 0.2       # confidence bonus when a sheet of paper is at the wrists

    # --- temporal alerting (both modes) ---
    persistence_s: float = 3.0              # "alert only if behaviour persists > 3 s"
    event_persistence_s: dict = field(default_factory=lambda: {"material_exchange": 1.0})  # exchanges are short

    def persistence_for(self, event_type):
        return self.event_persistence_s.get(event_type, self.persistence_s)

    # --- improvement only ---
    assign_phone_to_owner: bool = True      # each phone belongs to ONE nearest person
    gap_tolerance_s: float = 1.0            # detector drop-outs shorter than this don't end an episode
    min_positive_ratio: float = 0.6         # share of the episode the signal must actually be on
    min_alert_confidence: float = 0.0       # optionally suppress low-confidence alerts

    def to_dict(self):
        return asdict(self)


def load_config(path=None) -> Config:
    """Config with thresholds from a frozen JSON file (keys = Config fields), or defaults."""
    import json
    from pathlib import Path
    if not path:
        return Config()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Config(**{k: v for k, v in raw.items() if k in Config.__dataclass_fields__})
