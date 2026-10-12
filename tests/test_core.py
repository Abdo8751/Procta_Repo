"""Unit tests for alerting policies, phone attribution and evaluation.  Run: python -m pytest -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from procta.config import Config  # noqa: E402
from procta.evaluation import evaluate  # noqa: E402
from procta.signals import SignalExtractor, assign_phone_owners  # noqa: E402
from procta.temporal import BaselinePersistence, ConfidenceAwareAlerter  # noqa: E402

FPS = 10
K = (1, "phone_in_hand")


def feed(policy, pattern):
    """pattern: list of 0/1 per frame at FPS."""
    alerts = []
    for i, on in enumerate(pattern):
        alerts += policy.step(i / FPS, {K: 0.8} if on else {}, {1})
    return alerts


def test_both_alert_on_clean_episode_once():
    pat = [1] * 60 + [0] * 30
    for P in (BaselinePersistence, ConfidenceAwareAlerter):
        a = feed(P(Config()), pat)
        assert len(a) == 1 and abs(a[0].t_alert - 3.0) < 0.15


def test_no_alert_for_short_glance():
    for P in (BaselinePersistence, ConfidenceAwareAlerter):
        assert feed(P(Config()), [1] * 20 + [0] * 40) == []


def test_single_dropout_fragments_baseline_but_not_improved():
    # 8 s of phone use with one missed frame every 2.5 s
    pat = [1 if i % 25 else 0 for i in range(1, 81)] + [0] * 30
    assert len(feed(BaselinePersistence(Config()), pat)) == 0       # timer keeps resetting
    a = feed(ConfidenceAwareAlerter(Config()), pat)
    assert len(a) == 1 and 0.9 < a[0].positive_ratio <= 1.0 and a[0].confidence > 0.7


def test_long_gap_ends_episode():
    pat = [1] * 40 + [0] * 20 + [1] * 40      # 2 s gap > 1 s tolerance -> two episodes
    assert len(feed(ConfidenceAwareAlerter(Config()), pat)) == 2


def test_mostly_off_signal_does_not_alert():
    pat = [1, 0, 0, 0, 0] * 30                 # on 20 % of the time
    assert feed(ConfidenceAwareAlerter(Config()), pat) == []


def _person(pid, box, wrist):
    kps = [[0, 0, 0]] * 17
    kps = [list(k) for k in kps]
    kps[9] = [wrist[0], wrist[1], 0.9]
    kps[5], kps[6] = [box[0] + 20, box[1] + 30, 0.9], [box[2] - 20, box[1] + 30, 0.9]
    return {"id": pid, "box": box, "conf": 0.9, "kps": kps}


def test_phone_attributed_to_one_owner():
    a = _person(1, [0, 0, 200, 400], (150, 300))
    b = _person(2, [400, 0, 600, 400], (450, 300))
    phone = {"box": [140, 290, 170, 330], "conf": 0.7}
    assert assign_phone_owners([phone], [a, b]) == [0]
    frame = {"t": 0.0, "persons": [a, b], "phones": [phone]}
    base = SignalExtractor(Config(), "baseline")(frame)
    imp = SignalExtractor(Config(), "improved")(frame)
    assert (1, "phone_in_hand") in base and (2, "phone_on_desk") in base     # baseline bug
    assert (1, "phone_in_hand") in imp and (2, "phone_on_desk") not in imp


def _face_person(pid, nose_dx, hide_eye=None):
    """Person facing the camera; nose_dx shifts the nose sideways (eye distance = 40 px)."""
    p = _person(pid, [0, 0, 200, 400], (150, 300))
    p["kps"][1], p["kps"][2] = [80, 60, 0.9], [120, 60, 0.9]          # eyes
    p["kps"][0] = [100 + nose_dx, 75, 0.9]                            # nose
    if hide_eye is not None:
        p["kps"][hide_eye][2] = 0.05
    return p


def _run_head(sig, frames):
    return [sig({"t": i / FPS, "persons": [p], "phones": []}) for i, p in enumerate(frames)]


def test_head_turn_flagged_after_calibration_only():
    cfg = Config()
    sig = SignalExtractor(cfg, "improved")
    calib = [_face_person(1, 0)] * cfg.calibration_frames
    out = _run_head(sig, calib + [_face_person(1, 2), _face_person(1, 16), _face_person(1, -16)])
    assert all((1, "head_turned") not in o for o in out[:-2])        # calibration + facing forward
    assert (1, "head_turned") in out[-2] and (1, "head_turned") in out[-1]


def test_head_yaw_relative_to_calibration_and_profile():
    cfg = Config()
    sig = SignalExtractor(cfg, "improved")
    calib = [_face_person(1, 12)] * cfg.calibration_frames            # camera off to one side
    out = _run_head(sig, calib + [_face_person(1, 12), _face_person(1, 0, hide_eye=2)])
    assert (1, "head_turned") not in out[-2]                          # normal pose for this student
    assert (1, "head_turned") in out[-1]                              # profile: one eye hidden


def test_head_turn_alerts_once_after_3s_not_for_glance():
    pat = [1] * 20 + [0] * 30 + [1] * 50 + [0] * 20                   # 2 s glance, then 5 s turn
    alerts = []
    pol = ConfidenceAwareAlerter(Config())
    for i, on in enumerate(pat):
        alerts += pol.step(i / FPS, {(1, "head_turned"): 0.9} if on else {}, {1})
    assert len(alerts) == 1 and alerts[0].event_type == "head_turned"
    assert alerts[0].message("10:00:00") == "Potential sustained head turn detected at 10:00:00 - please review."


def test_evaluation_counts():
    gt = [{"event_type": "phone_in_hand", "start_s": 10, "end_s": 20},
          {"event_type": "phone_in_hand", "start_s": 40, "end_s": 41}]      # too short = not a target
    alerts = [{"event_type": "phone_in_hand", "t_alert": 13},
              {"event_type": "phone_in_hand", "t_alert": 18},               # duplicate
              {"event_type": "phone_in_hand", "t_alert": 40.5}]             # false alarm
    m = evaluate(alerts, gt, 60, 3.0, 1.0, ["phone_in_hand"])["phone_in_hand"]
    assert m["recall"] == 1.0 and abs(m["precision"] - 2 / 3) < 1e-9
    assert m["alerts_per_episode"] == 2.0 and m["false_alarms_per_min"] == 1.0
    assert m["mean_latency_s"] == 3.0


def test_frame_level_alerts_on_every_flicker():
    from procta.temporal import FrameLevelAlerter
    pat = [1, 1, 0, 1, 1, 0, 1] + [0] * 5 + [1]        # 3 runs then a glance -> 4 alerts
    assert len(feed(FrameLevelAlerter(Config()), pat)) == 4


def test_alert_message_is_not_a_verdict():
    from procta.temporal import Alert
    m = Alert(7, "phone_in_hand", 12.0, 9.0).message("10:43:21")
    assert m == "Potential mobile-phone interaction detected at 10:43:21 - please review."
    assert "cheat" not in m.lower()


def test_alert_log_records_detailed_alert(tmp_path):
    from datetime import datetime
    from procta.alert_log import AlertLog, connect
    from procta.temporal import Alert
    log = AlertLog(tmp_path / "a.db", "live_x", "improved", datetime(2026, 10, 12, 13, 50, 0))
    log.add(Alert(1, "head_turned", 13.0, 10.0, 0.915, 1.0), "live_x/evidence/e.jpg")
    log.close()
    r = dict(connect(tmp_path / "a.db").execute("SELECT * FROM alerts").fetchone())
    assert r["timestamp"] == "2026-10-12T13:50:13" and r["alert_name"] == "Head turned away"
    assert r["confidence_level"] == "High" and r["duration_s"] == 3.0 and r["review"] is None
    assert "13:50:13" in r["message"]
