# PROCTA – Intelligent Exam Proctor (decision support)

Detects a small set of predefined exam-room events in staged, consented mock-exam video
and produces **reviewable, time-stamped, confidence-aware alerts** for a human invigilator.
The system never concludes that a student cheated.

Events: `phone_in_hand`, `phone_on_desk`, `body_movement`, `unusual_seat_movement`, `head_turned`
(definitions and non-examples: [docs/event_definitions.md](docs/event_definitions.md)).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                     # 14 tests
```
YOLOv8 weights download automatically on first run (or put them in `models/`).

## Workflow (week 5: robot route, evidence clips, exam context)
```bash
# 0. Describe the mock exam and the robot's observation route
#    configs/exam_context.json  (permitted devices, neighbour side / clock side per point, ignore zones)
#    docs/annotation_guide.md   (event definitions, lookalikes, room set-up, recording plan)

# 1. Live demo - the camera plays the robot's camera; n/p = move to next/previous observation point
python scripts/live.py --mode context --auto-route        # proposed: temporal + consolidation + context
python scripts/live.py --mode improved                    # temporal + consolidation, no context
python scripts/live.py --mode frame                       # B0 frame-level baseline
python scripts/dashboard.py                               # review page: http://localhost:8765

# 2. Record staged sessions (solo protocol: phone + viewing neighbour's paper, with lookalikes)
python scripts/record_session.py --name s01               # uses docs/protocol_v2_solo.csv
python scripts/label_video.py --video data/videos/s01.mp4
python scripts/extract_detections.py --video data/videos/s01.mp4

# 3. One session, all four methods
python scripts/run_offline.py --detections data/detections/s01.jsonl --gt data/labels/s01_gt.csv --point A

# 4. All sessions, dev/test split (data/splits.csv), thresholds frozen before testing
python scripts/evaluate_sessions.py --set dev
python scripts/evaluate_sessions.py --freeze configs/thresholds_frozen.json
python scripts/evaluate_sessions.py --set test --thresholds configs/thresholds_frozen.json

# 5. Group recording (6 students, fixed phone camera): convert once, draw seats, extract
python scripts/convert_video.py "C:/Users/Ar/Downloads/Test_1.MOV" data/videos/Test_1.mp4   # 4K HEVC -> 1080p H.264
python scripts/define_seats.py --video data/videos/Test_1.mp4 --auto --t 60    # seat zones F1-F3, B1-B3 -> point T1
python scripts/extract_detections.py --video data/videos/Test_1.mp4 --stride 2 --imgsz 960 --det-imgsz 1280   # ~15 min on CPU
python scripts/live.py --mode context --video data/videos/Test_1.mp4 --start-point T1

# Replay any recording as if live (demo with a different input)
python scripts/live.py --mode context --video data/videos/s04.mp4
```

Methods compared on identical detections: **B0** frame-level (`frame`), **B1** 3-s rule (`baseline`),
**temporal confirmation + consolidation** (`improved`), **+ examination context** (`context`, the focused
improvement). Every confirmed event is one record with timestamp, observation point, image region, a still,
one evidence clip per sighting (animated WebP + MP4) and the invigilator's Confirm / Dismiss decision.

## Layout
```
procta/       config, detect (YOLO), signals (per-frame events), temporal (B0/B1/temporal alerting), events (consolidation),
              context (exam rules + route), seats (seat zones -> identity), clips (rolling-buffer evidence clips), alert_log (SQLite),
              evaluation (event-level P/R/false alarms/latency), review (proctor review page), draw
scripts/      live, dashboard, record_session, label_video, extract_detections, run_offline, evaluate_sessions,
              convert_video, define_seats,
              simulate_detector_noise, analyze_live_log
configs/      exam_context.json (rules + observation route), thresholds_frozen.json (after --freeze)
docs/         event definitions, recording protocol, week-2 report
data/labels/  ground truth CSVs (videos are git-ignored)
results/      metrics and simulation outputs
legacy/       (optional) drop the original procta_live_baseline.py here for history
```

## Metrics (event level)
Research question: can temporal and contextual analysis improve reliable detection of predefined
exam-room events compared with simple frame-level detection?

precision = true alerts / alerts · recall = detected episodes / episodes ≥ 3 s · missed-event rate = 1 − recall ·
alerts per exam ·
false alarms per minute · latency = first alert − episode start ·
alerts per episode (duplicate alerts = extra review effort) · human review time (review page export).

## Privacy
Staged mock exams only, with written consent and ethics approval; never a real exam.
Raw video and evidence images are git-ignored and stay on the project drive; only labels and aggregate metrics are committed.
