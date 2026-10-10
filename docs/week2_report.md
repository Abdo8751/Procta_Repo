# PROCTA – Week 2 progress: baseline, first results, improvement

**Main research question (agreed with supervisor)**
*Can temporal and contextual analysis improve the reliable detection of predefined exam-room events compared with simple frame-level detection?*

Compared systems: **B0 frame-level** (event seen in a frame → alert), **B1 3-s persistence** (first working version, rule from meeting 2), **PROCTA improved** (temporal + contextual, below).

Sub-questions from the project description:
- **RQ1** Which predefined exam-room events can be detected with high enough precision/recall to support invigilators without excessive false alarms?
- **RQ2** Does evidence-rich, confidence-aware alerting reduce human review effort while preserving human judgement?

## 1. Baselines (implemented)
**B0 frame-level** (`FrameLevelAlerter`): same per-frame signals, an alert every time a signal switches on, no time threshold — the "phone detected → alert" baseline expected to give many false alarms.

**B1 3-s persistence** (the live webcam version):
Webcam pipeline: YOLOv8n (phone, COCO class 67) + YOLOv8n-pose (people, ByteTrack ids, 17 keypoints).
Per-frame rules → 4 events: `phone_in_hand` (phone centre within 1.5× phone diagonal of a wrist),
`phone_on_desk` (phone present, not near a wrist), `body_movement` (shoulder-centre speed > 150 px/s),
`unusual_seat_movement` (person centre > 80 px from a 30-frame calibrated seat position).
Temporal rule: alert once a signal has been on for 3 s **with no off frame**; one alert per run.
Code: `procta/signals.py` (mode `baseline`) + `BaselinePersistence` in `procta/temporal.py`.

## 2. Measurable baseline results so far

### 2a. Live log (5 webcam sessions, 39 alerts) – `results/live_log_analysis.md`
No ground truth was recorded for these sessions, so these are observations, not precision/recall:
- **41 % of alerts (16/39) are re-fires** of the same track + event within 20 s (median gap 16 s, min 4 s).
  Either the behaviour genuinely restarted, or one continuous episode was split by a missed detection.
- **4 seconds with alerts on two different track ids at once** (tracks 2+5 `phone_in_hand`; tracks 2+29
  in-hand + on-desk). With one phone, this is either a tracker ID switch or the same phone being attributed to two people.
- 4 different track ids (2, 4, 5, 29) appear across sessions → identity is not stable.

### 2b. Controlled detector-noise simulation – `results/simulation/`
Known ground truth, 20 × 10-min sessions per point, 12 fps, bursty detector misses (mean drop-out 0.4 s),
25 % of episodes are < 3 s "glances" that must not alert, ~1.5 false-positive bursts/min.

| per-frame miss rate | policy | precision | recall | false alarms/min | latency (s) | alerts/true episode |
|---|---|---|---|---|---|---|
| 0.10 | B0 frame | 0.82 | 1.00 | **1.41** | 0.20 | **6.64** |
| 0.20 | B0 frame | 0.88 | 1.00 | **1.58** | 0.09 | **12.22** |
| 0.10 | B1 persistence | 0.95 | 0.94 | 0.12 | 5.05 | **2.62** |
| 0.10 | improved | 0.92 | 0.98 | 0.12 | 3.12 | **1.27** |
| 0.20 | B1 persistence | 0.95 | **0.77** | 0.07 | 7.96 | 1.90 |
| 0.20 | improved | 0.95 | **0.99** | 0.07 | 3.37 | 1.38 |
| 0.30 | B1 persistence | 0.80 | **0.42** | 0.14 | 13.34 | 1.35 |
| 0.30 | improved | 0.91 | **0.97** | 0.15 | 4.67 | 1.51 |

(Full table and gap-tolerance ablation: `results/simulation/summary.md`, plot `sweep.png`.)
Reading: B0 finds every episode but raises ~12× more false alarms (every < 3 s glance alerts) and 6–12 alerts per real episode → huge review load. B1 cuts false alarms but loses recall as soon as the detector flickers. The improved policy keeps B1's false-alarm rate with B0-like recall and ~1.3 alerts per episode.
Note: alert-level precision rewards duplicates (B0's duplicates inside true episodes count as "true"), so alerts/episode and false alarms/min are the fairer comparison. B1 alerts/episode falls at high miss rates only because it detects almost nothing.

### 2c. Real staged-video benchmark – **to fill after recording**
| session | policy | precision | recall / missed-event rate | false alarms/min | alerts per exam | alerts/episode |
|---|---|---|---|---|---|---|
| s01 | B0 frame | | | | | |
| s01 | B1 persistence | | | | | |
| s01 | improved | | | | | |

## 3. Baseline limitations
1. **Brittle persistence** – one missed frame resets the 3 s timer → long episodes fragment into many alerts (review burden) and flickery episodes never alert (missed detections, high latency). Supported by 2a (41 % re-fires) and 2b.
2. **Phone attribution bug** – every phone not near person A's wrist is `phone_on_desk` *for A*, so one phone in B's hand also flags A (matches the track 2 + 29 log entry). Unit test `test_phone_attributed_to_one_owner` reproduces it.
3. **No confidence** – all alerts look equally certain; an invigilator cannot triage.
4. **No evidence** – the log has a timestamp only; nobody can verify an alert afterwards.
5. **Wall-clock timestamps, no recorded video** → alerts cannot be matched to ground truth.
6. **Fixed pixel thresholds** (150 px/s, 80 px) depend on camera distance/resolution; tracker ID switches reset per-person state.
7. YOLOv8n COCO "cell phone" is weak for small/occluded phones (held low, edge-on); persistence cannot fix detector recall.

## 4. Proposed improvement (linked to RQ2, partly RQ1): confidence-aware, evidence-rich alerting
- **Gap-tolerant episodes**: drop-outs < 1 s do not end an episode; alert when episode ≥ 3 s and signal on ≥ 60 % of it → one alert per real episode.
- **Confidence score** per alert = mean detector confidence × positive ratio → alerts can be ranked/thresholded.
- **Phone owner assignment**: each phone belongs to the nearest person only; far-away phones ignored.
- **Keypoint-confidence gating** for wrists/shoulders.
- **Wording**: every alert is a review prompt, e.g. "Potential mobile-phone interaction detected at 10:43:21 – please review." — never a verdict.
- **Evidence**: side-by-side image (episode start | alert moment) with boxes, plus a review page where the invigilator must Confirm/Dismiss each alert; decisions and **seconds per decision** are exported → direct measure of review effort for RQ2.

**Status: implemented** (`ConfidenceAwareAlerter`, `assign_phone_owners`, `scripts/run_offline.py`, `procta/review.py`), unit-tested (9 tests), validated in simulation; evaluation on real staged video is the next step.

## 5. On the simulator
PROCTA's core is perception on video; the robot (Unitree) only matters for optional patrol between fixed observation points.
The controlled environment used now is a **scripted, consented mock-exam recording protocol** (`docs/protocol_v1.csv`) with ground truth known by design, plus the detector-noise simulation.
Proposal: a Webots or Isaac Sim scene (exam room + robot at predefined observation points, rendering the camera view into this same pipeline) once the perception benchmark is established — to discuss in the meeting.

## 6. Next steps
1. Record 2–3 staged sessions (different lighting/angle) with `record_session.py`; correct labels with `label_video.py`.
2. Run B0 vs B1 vs improved on them → fill table 2c (precision, recall, missed-event rate, false-alarm rate, alerts per exam); per-event precision/recall answers RQ1.
3. Human review time (metric from the supervisor's list) via the review page: same alert set reviewed with baseline list vs improved (ranked, with evidence) → review time and decision accuracy (RQ2).
4. Evidence clips (a few seconds of video) in addition to the evidence images.
5. Fine-tune the phone detector on own frames if phone recall is the bottleneck.
