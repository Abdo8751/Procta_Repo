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
