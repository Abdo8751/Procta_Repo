"""Central configuration. Every threshold used by the system lives here so that
experiments can state exactly which values produced which numbers."""

from dataclasses import dataclass, field, asdict

# COCO / COCO-keypoint indices
PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67          # 'cell phone'
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
NOSE, LEFT_EYE, RIGHT_EYE = 0, 1, 2

EVENT_TYPES = ("phone_in_hand", "phone_on_desk", "body_movement", "unusual_seat_movement",
               "head_turned")


@dataclass
class Config:
    # --- models / inference ---
    det_model: str = "models/yolov8n.pt"
    pose_model: str = "models/yolov8n-pose.pt"
    imgsz: int = 416
    phone_conf: float = 0.25          # minimum detector confidence for a phone box
    det_frame_skip: int = 1           # 1 = run phone detector on every frame (offline default)

    # --- per-frame event signals ---
    phone_proximity_factor: float = 1.5     # wrist-to-phone distance < factor * phone diagonal
    keypoint_min_conf: float = 0.3          # ignore wrists/shoulders the pose model is unsure about
    body_movement_speed: float = 150.0      # shoulder-centroid speed, px/s
    seat_drift_px: float = 80.0             # distance from calibrated seat centroid, px
    calibration_frames: int = 30
    head_yaw_thr: float = 0.25              # |yaw - calibrated yaw|, yaw = (nose_x - eye-mid x) / eye distance

    # --- temporal alerting (both modes) ---
    persistence_s: float = 3.0              # "alert only if behaviour persists > 3 s"

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
