"""Tests for the week-5 additions: exam context, consolidation, clips, robot observation log."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from procta.config import Config  # noqa: E402
from procta.context import ExamContext, ObservationPoint, in_zone  # noqa: E402
from procta.events import EventConsolidator  # noqa: E402
from procta.signals import SignalExtractor  # noqa: E402
from procta.temporal import Alert, run_policy  # noqa: E402
from test_core import _face_person, _person  # noqa: E402

FPS = 10


def _ctx(neighbour="left", allowed=("right",), zones=(), devices=()):
    pt = ObservationPoint("A", neighbour_side=neighbour, allowed_look_sides=list(allowed), ignore_zones=list(zones))
    return ExamContext(permitted_devices=list(devices), observation_points=[pt]), pt


def _head_frames(cfg, dx, n):
    return [_face_person(1, 0)] * cfg.calibration_frames + [_face_person(1, dx)] * n


def test_default_context_file_loads():
    ctx = ExamContext.load()
    assert len(ctx.observation_points) >= 1 and "cell phone" not in ctx.permitted_devices


def test_context_keeps_neighbour_turn_drops_clock_turn():
    cfg = Config()
    ctx, pt = _ctx(neighbour="left", allowed=("right",))
    for dx, expect in ((-16, True), (16, False)):          # nose to image left = towards neighbour
        sig = SignalExtractor(cfg, "context", context=ctx, point=pt)
        out = [sig({"t": i / FPS, "persons": [p], "phones": []}) for i, p in enumerate(_head_frames(cfg, dx, 5))]
        assert ((1, "head_turned") in out[-1]) is expect
    imp = SignalExtractor(cfg, "improved")                 # without context every turn counts
    out = [imp({"t": i / FPS, "persons": [p], "phones": []}) for i, p in enumerate(_head_frames(cfg, 16, 5))]
    assert (1, "head_turned") in out[-1]


def test_context_ignore_zone_and_permitted_device():
    person = _person(1, [0, 0, 200, 400], (150, 300))
    phone = {"box": [140, 290, 170, 330], "conf": 0.7}
    frame = {"t": 0.0, "wh": [640, 480], "persons": [person], "phones": [phone]}
    ctx, pt = _ctx(zones=[[0.0, 0.5, 0.5, 1.0]])           # phone centre (0.24, 0.64) is inside
    assert SignalExtractor(Config(), "context", context=ctx, point=pt)(frame) == {}
    ctx2, pt2 = _ctx(devices=["cell phone"])
    sig = SignalExtractor(Config(), "context", context=ctx2, point=pt2)
    assert sig(frame) == {} and sig.suppressed["phone permitted by exam rules"] == 1
    assert in_zone([0, 0, 10, 10], [], [640, 480]) is False


def test_consolidation_merges_repeats_within_window():
    c = EventConsolidator(window_s=30)
    e1, new1 = c.on_alert(Alert(1, "phone_in_hand", 13, 10, 0.8, 1.0), "A")
    c.on_episode_end((1, "phone_in_hand"), 18, "A")
    e2, new2 = c.on_alert(Alert(7, "phone_in_hand", 26, 23, 0.9, 1.0), "A")   # new track id, 5 s later
    e3, new3 = c.on_alert(Alert(7, "phone_in_hand", 90, 87, 0.9, 1.0), "A")   # 60 s later -> new event
    e4, new4 = c.on_alert(Alert(7, "phone_in_hand", 95, 92, 0.9, 1.0), "B")   # other point -> new event
    assert new1 and not new2 and e2 is e1 and e1.sightings == 2 and e1.track_ids == {1, 7}
    assert new3 and new4 and len(c.events) == 3 and e1.confidence == 0.9


def _phone_frames(pattern):
    """pattern of 0/1 per frame: phone in the right hand or no phone."""
    person = _person(1, [0, 0, 200, 400], (150, 300))
    phone = {"box": [140, 290, 170, 330], "conf": 0.7}
    return [{"t": i / FPS, "persons": [person], "phones": [phone] if on else []} for i, on in enumerate(pattern)]


def test_repeated_sighting_is_one_record_in_temporal_modes_two_in_b1():
    pat = [0] * 40 + [1] * 60 + [0] * 50 + [1] * 60 + [0] * 30          # 6 s, 5 s away, 6 s
    frames = _phone_frames(pat)
    cfg = Config()
    b1 = [a for a in run_policy(frames, "baseline", cfg) if a.event_type == "phone_in_hand"]
    imp, events, _ = run_policy(frames, "improved", cfg, return_events=True)
    imp = [a for a in imp if a.event_type == "phone_in_hand"]
    assert len(b1) == 2 and len(imp) == 1
    ev = next(e for e in events if e.event_type == "phone_in_hand")
    assert ev.sightings == 2


def test_aggregate_sums_sessions():
    from procta.evaluation import aggregate, evaluate
    gt = [{"event_type": "phone_in_hand", "start_s": 10, "end_s": 20}]
    r1 = evaluate([{"event_type": "phone_in_hand", "t_alert": 13}], gt, 1800, 3.0, 1.0, ["phone_in_hand"])
    r2 = evaluate([{"event_type": "phone_in_hand", "t_alert": 50}], gt, 1800, 3.0, 1.0, ["phone_in_hand"])
    a = aggregate([r1, r2], [1800, 1800])["phone_in_hand"]
    assert a["alerts"] == 2 and a["detected"] == 1 and a["target_episodes"] == 2
    assert a["false_alerts_per_hour"] == 1.0 and a["recall"] == 0.5


def test_alert_log_consolidation_and_observations(tmp_path):
    import json
    from datetime import datetime
    from procta.alert_log import AlertLog, connect
    log = AlertLog(tmp_path / "a.db", "live_x", "context", datetime(2026, 10, 22, 10, 0, 0))
    rid = log.add(Alert(1, "phone_in_hand", 13.0, 10.0, 0.8, 1.0), "e.jpg", "A", [0, 0, 10, 10], "c1.webp")
    log.update_event(rid, 2, 30.0, "c2.webp")
    log.observe(40.0, "A", "insufficient_view", "next permitted point B")
    log.close()
    con = connect(tmp_path / "a.db")
    r = dict(con.execute("SELECT * FROM alerts").fetchone())
    assert r["sightings"] == 2 and json.loads(r["clips"]) == ["c1.webp", "c2.webp"] and r["observation_point"] == "A"
    o = dict(con.execute("SELECT * FROM observations").fetchone())
    assert o["kind"] == "insufficient_view" and o["timestamp"] == "2026-10-22T10:00:40"


def test_clip_recorder_writes_preroll_clip(tmp_path):
    import numpy as np
    from procta.clips import ClipRecorder
    rec = ClipRecorder(tmp_path, pre_s=3, post_s=0.5, fps=5, width=64)
    for i in range(50):                        # 10 s at 5 fps
        t = i / 5
        rec.push(t, np.full((48, 64, 3), i * 5 % 255, np.uint8))
        if t == 4.0:
            rec.start("ev001_s1", t_from=2.0)  # confirmed at 4 s, behaviour started at 2 s
        if t == 6.0:
            rec.stop("ev001_s1", 6.0)
    rec.join()
    assert (tmp_path / "ev001_s1.mp4").exists()
    assert (tmp_path / "ev001_s1.webp").exists() or (tmp_path / "ev001_s1.gif").exists()
    import cv2
    n = int(cv2.VideoCapture(str(tmp_path / "ev001_s1.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    assert 25 <= n <= 30                        # from 1 s before the start (t=1.0) to stop + 0.5 s (t=6.5)


# ---- seats (Test_1: 6 people, tracker ids switch) ------------------------------------------

def _pe(pid, box):
    return {"id": pid, "box": box, "conf": 0.9, "kps": [[0, 0, 0]] * 17}


ZONES = {"F1": [0.0, 0.5, 0.33, 1.0], "F2": [0.33, 0.5, 0.66, 1.0], "B1": [0.0, 0.1, 0.33, 0.5],
         "invigilator": [0.9, 0.0, 1.0, 1.0]}


def test_seat_assignment_by_box_centre_and_invigilator():
    from procta.seats import INVIGILATOR, assign_seats
    wh = [1000, 1000]
    persons = [_pe(7, [50, 550, 300, 950]),       # front left
               _pe(9, [400, 600, 600, 950]),      # front middle
               _pe(3, [60, 150, 300, 450]),       # back left
               _pe(4, [920, 100, 990, 900]),      # in the invigilator zone
               _pe(5, [700, 100, 800, 600])]      # between the rows, outside all seats -> walking
    assert assign_seats(persons, ZONES, wh) == ["F1", "F2", "B1", INVIGILATOR, INVIGILATOR]
    assert assign_seats(persons, {}, wh) == [None] * 5                      # no zones: track ids


def test_standing_person_in_front_of_seat_is_invigilator():
    from procta.seats import INVIGILATOR, assign_seats
    tall = _pe(8, [380, 0, 600, 1000])            # centre in F2 zone but twice its height
    seated = _pe(2, [400, 600, 600, 950])
    assert assign_seats([tall, seated], ZONES, [1000, 1000]) == [INVIGILATOR, "F2"]


def test_seat_identity_survives_track_id_switch():
    from procta.context import ObservationPoint
    from procta.signals import SignalExtractor
    from procta.temporal import ConfidenceAwareAlerter
    cfg = Config()
    sig = SignalExtractor(cfg, "improved", point=ObservationPoint("T1", seat_zones=ZONES))
    pol, alerts = ConfidenceAwareAlerter(cfg), []
    for i in range(60):                           # same student, tracker id changes every 10 frames
        p = _pe(100 + i // 10, [50, 550, 300, 950])
        p["kps"] = [list(k) for k in p["kps"]]
        p["kps"][9] = [150, 800, 0.9]
        fr = {"t": i / 10, "wh": [1000, 1000], "persons": [p], "phones": [{"box": [140, 790, 170, 830], "conf": 0.8}]}
        alerts += pol.step(fr["t"], sig(fr), sig.present)
    assert [(a.track_id, a.event_type) for a in alerts] == [("F1", "phone_in_hand")]


def test_consolidation_keeps_seats_apart():
    from procta.events import EventConsolidator
    from procta.temporal import Alert
    c = EventConsolidator(30)
    _, n1 = c.on_alert(Alert("F1", "phone_in_hand", 10, 7, 0.8, 1), "T1")
    _, n2 = c.on_alert(Alert("B2", "phone_in_hand", 12, 9, 0.8, 1), "T1")   # other student: new event
    _, n3 = c.on_alert(Alert("F1", "phone_in_hand", 25, 22, 0.8, 1), "T1")  # same seat again: merged
    assert (n1, n2, n3) == (True, True, False)


# ---- Phase 2: phone vs calculator, looking_down, material_exchange ---------------------------

def _body(pid, cx, top, sw=100, wrists=((0.3, 1.6), (-0.3, 1.6)), wrist_conf=0.9, pitch=0.0):
    """Seated person facing the camera: shoulders at y=top+150, shoulder width sw.
    wrists: (dx, dy) in shoulder widths from the shoulder centre. pitch: ears raised by pitch x sw."""
    k = [[0, 0, 0.0] for _ in range(17)]
    sy = top + 150
    k[5], k[6] = [cx + sw / 2, sy, 0.9], [cx - sw / 2, sy, 0.9]          # shoulders
    k[1], k[2] = [cx + 15, top + 60, 0.9], [cx - 15, top + 60, 0.9]      # eyes
    k[0] = [cx, top + 75, 0.9]                                           # nose
    k[3], k[4] = [cx + 35, top + 60 - pitch * sw, 0.9], [cx - 35, top + 60 - pitch * sw, 0.9]   # ears
    for i, (dx, dy) in zip((9, 10), wrists):
        k[i] = [cx + dx * sw, sy + dy * sw, wrist_conf]
    return {"id": pid, "box": [cx - 150, top, cx + 150, top + 450], "conf": 0.9, "kps": k}


def _frames(make, n, objects=(), wh=(1000, 1000)):
    return [{"t": i / 10, "wh": list(wh), "persons": make(i), "phones": [], "objects": list(objects)} for i in range(n)]


def test_clip_decision_calculator_is_not_a_phone():
    from procta.verify import decide
    assert decide({"a smartphone": 0.01, "a calculator": 0.98, "a pencil case": 0.01}) == ("calculator", 0.98)
    assert decide({"a smartphone": 0.69, "a calculator": 0.14, "a pencil case": 0.1}) == ("cell phone", 0.69)
    assert decide({"a smartphone": 0.3, "a calculator": 0.28, "a pencil case": 0.25}) is None       # unsure: dropped
    assert decide({"a smartphone": 0.03, "a calculator": 0.26, "a pencil case": 0.67}) is None      # pencil case


def test_calculator_in_hand_gives_no_phone_event_and_is_logged_in_context():
    calc = {"cls": "calculator", "box": [130, 290, 170, 330], "conf": 0.98}
    pe = _person(1, [0, 0, 200, 400], (150, 300))
    for mode in ("improved", "context"):
        sig = SignalExtractor(Config(), mode, context=ExamContext(permitted_devices=["calculator"], observation_points=[ObservationPoint("A")]))
        out = sig({"t": 0.0, "persons": [pe], "phones": [], "objects": [calc]})
        assert not any(ev.startswith("phone") for _, ev in out)
        assert bool(sig.notes) == (mode == "context")


def test_writing_pose_is_not_looking_down_but_lap_look_is():
    cfg = Config()
    calib = cfg.calibration_frames
    # writing: head lowered a lot (pitch 0.2) but hands stay on the desk -> never looking_down
    sig = SignalExtractor(cfg, "improved")
    fr = _frames(lambda i: [_body(1, 500, 100, pitch=0.0 if i < calib else 0.2)], calib + 40)
    assert not any((1, "looking_down") in sig(f) for f in fr)
    # head lowered AND hands gone below the desk edge (wrists not visible) -> looking_down
    sig = SignalExtractor(cfg, "improved")
    fr = _frames(lambda i: [_body(1, 500, 100, pitch=0.0 if i < calib else 0.2,
                                  wrist_conf=0.9 if i < calib else 0.05)], calib + 40)
    outs = [sig(f) for f in fr]
    assert all((1, "looking_down") not in o for o in outs[:calib])
    assert all((1, "looking_down") in o for o in outs[calib + 1:])
    assert Alert(1, "looking_down", 5, 2).message("10:00:00") == \
        "Potential looking down below desk detected at 10:00:00 - please review."


ZONES2 = {"F1": [0.0, 0.0, 0.5, 1.0], "F2": [0.5, 0.0, 1.0, 1.0]}


def _pair(reach):
    """Two neighbours (F1 at x=250, F2 at x=750); with reach their inner wrists meet at x~500."""
    a = _body(1, 250, 100, sw=150, wrists=((1.55, 0.8), (-0.3, 1.6)) if reach else ((0.3, 1.6), (-0.3, 1.6)))
    b = _body(2, 750, 100, sw=150, wrists=((0.3, 1.6), (-1.55, 0.8)) if reach else ((0.3, 1.6), (-0.3, 1.6)))
    return [a, b]


def test_material_exchange_between_two_seats_alerts_after_1s():
    from procta.temporal import ConfidenceAwareAlerter
    cfg = Config()
    pt = ObservationPoint("T1", seat_zones=ZONES2)
    sig, pol = SignalExtractor(cfg, "improved", point=pt), ConfidenceAwareAlerter(cfg)
    paper = {"cls": "paper", "box": [480, 340, 520, 380], "conf": 0.5}
    fr = _frames(lambda i: _pair(reach=40 <= i < 55), 70, objects=[paper])      # 1.5 s hand-over
    alerts = []
    for f in fr:
        s = sig(f)
        if 40 <= f["t"] * 10 < 55:
            assert ("F1+F2", "material_exchange") in s
        alerts += pol.step(f["t"], s, sig.present)
    ex = [a for a in alerts if a.event_type == "material_exchange"]
    assert len(ex) == 1 and ex[0].track_id == "F1+F2" and 4.9 < ex[0].t_alert < 5.2
    sig2 = SignalExtractor(cfg, "improved", point=pt)
    assert not any(("F1+F2", "material_exchange") in sig2(f) for f in _frames(lambda i: _pair(False), 30))


def test_exchange_with_invigilator_is_logged_not_signalled_in_context_mode():
    zones = {"F1": [0.0, 0.0, 0.5, 1.0], "invigilator": [0.5, 0.0, 1.0, 1.0]}
    pt = ObservationPoint("T1", seat_zones=zones)
    fr = _frames(lambda i: _pair(reach=True), 5)
    imp = SignalExtractor(Config(), "improved", point=pt)
    assert any(("F1+INV", "material_exchange") in imp(f) for f in fr)
    ctx = SignalExtractor(Config(), "context", context=ExamContext(), point=pt)
    assert not any(ev == "material_exchange" for f in fr for _, ev in ctx(f))
    assert ctx.suppressed["authorised exchange with invigilator"] > 0 and ctx.notes[0][2] == "authorised_exchange"
