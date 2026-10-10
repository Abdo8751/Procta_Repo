# PROCTA – Intelligent Exam Proctor (decision support)

Detects a small set of predefined exam-room events in staged, consented mock-exam video
and produces **reviewable, time-stamped, confidence-aware alerts** for a human invigilator.
The system never concludes that a student cheated.

Events: `phone_in_hand`, `phone_on_desk`, `body_movement`, `unusual_seat_movement`
(definitions and non-examples: [docs/event_definitions.md](docs/event_definitions.md)).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                     # 7 tests
```
YOLOv8 weights download automatically on first run (or put them in `models/`).

## Workflow
```bash
# 1. Record a staged session that follows the timed protocol (ground truth by design)
python scripts/record_session.py --name s01_daylight

# 2. Check / correct the ground truth on the video
python scripts/label_video.py --video data/videos/s01_daylight.mp4

# 3. Run the detectors once, cache detections
python scripts/extract_detections.py --video data/videos/s01_daylight.mp4

# 4. Baseline vs improved: alerts, metrics, evidence images, review pages
python scripts/run_offline.py --detections data/detections/s01_daylight.jsonl \
    --gt data/labels/s01_daylight_gt.csv --video data/videos/s01_daylight.mp4
#    -> results/s01_daylight/metrics.md, review_improved.html

# Live demo (baseline mode reproduces the original script)
python scripts/live.py --mode improved
```
Other: `scripts/simulate_detector_noise.py` (controlled detector-flicker experiment),
`scripts/analyze_live_log.py` (stats on the original live log).

## Layout
```
procta/       config, detect (YOLO), signals (per-frame events), temporal (baseline + improved alerting),
              evaluation (event-level P/R/false alarms/latency), review (proctor review page), draw
scripts/      record_session, label_video, extract_detections, run_offline, live, simulate_detector_noise, analyze_live_log
docs/         event definitions, recording protocol, week-2 report
data/labels/  ground truth CSVs (videos are git-ignored)
results/      metrics and simulation outputs
legacy/       (optional) drop the original procta_live_baseline.py here for history
```

## Metrics (event level)
precision = true alerts / alerts · recall = detected episodes / episodes ≥ 3 s ·
false alarms per minute · latency = first alert − episode start ·
alerts per episode (duplicate alerts = extra review effort).

## Privacy
Staged mock exams only, with written consent and ethics approval; never a real exam.
Raw video and evidence images are git-ignored and stay on the project drive; only labels and aggregate metrics are committed.
