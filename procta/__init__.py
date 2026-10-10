"""PROCTA - decision-support proctoring: detect predefined exam-room events and
produce reviewable, time-stamped, confidence-aware alerts for a human invigilator.
The system never concludes that a student cheated; every alert is a prompt for review."""

from .config import Config, EVENT_TYPES  # noqa: F401
