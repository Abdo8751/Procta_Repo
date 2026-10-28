# AI use log (Claude Code)

The thesis must disclose AI assistance. This log lists what Claude Code (Anthropic, model
Claude Opus 5.5, run inside VS Code) wrote or changed in this repository, per file. Design
decisions were proposed by the AI and accepted, checked and tested by the author; all numbers
in the report come from running the code on the author's own recordings.

Earlier work (before this log was started) was also AI-assisted: the week-2 refactor and the
week-5 patch were generated with an AI assistant and applied by the author; in Claude Code
sessions the AI added the head-turn event, the SQLite alert log + dashboard (incl. delete
buttons) and the webcam/pose-skeleton update (git history: a976cbb, acd7f8f, 1a4efea, ebfda62).

## 2026-10-28 - Phase 1: make the group video (Test_1) processable

| File | What the AI did |
|---|---|
| `scripts/convert_video.py` | New. Converts the 4K HEVC phone .MOV once to 1080p H.264 .mp4 with the ffmpeg bundled in `imageio-ffmpeg` (no system ffmpeg on the laptop); drops audio. |
| `scripts/extract_detections.py` | Added `--stride`, `--imgsz`, `--det-imgsz`, `--start`, `--seconds`; skipped frames are only grabbed and processed frames keep their real timestamps; prints speed and an estimate for the whole video. |
| `procta/config.py` | Added `det_imgsz` (separate object-detector input size) and `standing_ratio` (seat assignment). |
| `procta/detect.py` | Object detector uses `det_imgsz` when set. |
| `procta/seats.py` | New. Assigns each person a seat from normalised seat zones (box centre, best overlap, one person per seat); standing / outside-all-seats / invigilator-zone persons become `INV`. |
| `procta/signals.py` | Identity of a person = seat label when the observation point has seat zones, else the tracker id; the invigilator gets no examinee events; exposes `present` identities. |
| `procta/temporal.py` | `run_policy` passes the extractor's present identities to the policy (plumbing only; alerting logic unchanged). |
| `procta/events.py` | Consolidation only merges sightings of the same seat when identities are seats (previously type + point only, which would merge different students). |
| `procta/context.py` | `ObservationPoint` gained `seat_zones` and `on_route`. |
| `procta/alert_log.py` | New `seat` column; identity may be a seat label. |
| `procta/draw.py` | Draws seat zones and shows `seat (id n)` on each person. |
| `scripts/live.py` | Route cycles only `on_route` points (fixed viewpoints via `--start-point`); alerts, evidence names and the yaw overlay use the seat identity. |
| `scripts/define_seats.py` | New. Click/drag tool for seat zones, plus an automatic draft from YOLOv8-pose boxes (`--auto`). |
| `configs/exam_context.json` | Seats F1-F3/B1-B3 with rows and neighbours; new fixed point `T1` with the auto-drafted seat zones (to be checked by the author). |
| `tests/test_week5.py` | 4 tests: seat assignment + invigilator, standing person, seat identity across tracker id switches, consolidation keeps seats apart. |
| `README.md`, `requirements.txt` | Group-video workflow; `imageio-ffmpeg`. |

Measurements made by the AI on the author's laptop (CPU only): see the Phase 1 summary in the
commit message / chat (0.23 s per processed frame at pose 960, detector 1280, stride 2).

## 2026-10-28 - Replay controls and dismiss = delete evidence (author's requests)

| File | What the AI did |
|---|---|
| `scripts/live.py` | Video replay: `--stride`, `--det-imgsz`; keys space (pause), j/l (-/+10 s), ,/. (-/+2 s), +/- (speed = analyse every Nth frame). A jump ends running episodes, clears motion history and clip buffers, so re-watched behaviour merges into its existing event. |
| `procta/alert_log.py` | `purge_evidence()`: deletes an alert's still + clips (both formats) inside `results/` only and blanks the paths; the row and the decision stay. |
| `scripts/dashboard.py` | Dismiss and Delete purge the evidence files; dismissed rows are hidden (checkbox "Show dismissed"); "Seat / track" column. Design: dismissed decisions are kept (not deleted) so accuracy = confirmed / reviewed is not inflated to 100 %. |
| `tests/test_core.py` | Test: dismiss purges files, keeps the decision, never touches files outside `results/`. |

`configs/exam_context.json`: seat zones redrawn by the author with `define_seats.py` (not AI).

## 2026-10-28 - Phase 2: new / improved detection

| File | What the AI did |
|---|---|
| `procta/verify.py` | New. CLIP (ViT-B/32) zero-shot crop classifier (smartphone / calculator / pencil case / pen / paper / hand), batched per frame, cached for boxes that did not move; `decide()` rule. |
| `procta/detect.py` | `object_detector="world"`: YOLO-World (yolov8s-worldv2, classes cell phone / calculator / sheet of paper / book) proposes objects, CLIP verifies every phone/calculator box; output `phones` (verified) + `objects`. `"coco"` unchanged. |
| `procta/config.py` | `object_detector`, `world_*`, `clip_phone_min`; `looking_down` and `material_exchange` thresholds; `seat_drift_rel`; `event_persistence_s` (exchange = 1 s) + `persistence_for()`; new EVENT_TYPES. |
| `procta/signals.py` | `looking_down` (ear-vs-eye pitch beyond the calibrated writing pose AND wrists hidden or lowered), `material_exchange` (wrists of two different seats meet between them; paper bonus; with the invigilator: note in context mode), calculators never give phone events (context: permitted-device note), seat drift relative to the calibrated box width. |
| `procta/temporal.py` | Persistence threshold looked up per event type (`cfg.persistence_for`); new neutral messages. Episode/alert logic unchanged. |
| `procta/alert_log.py`, `scripts/dashboard.py` | Names for the new events; observation kinds `permitted_device`, `authorised_exchange`. |
| `procta/draw.py`, `scripts/live.py` | Calculators drawn in orange; pitch readout under the yaw readout; `--object-detector`; context notes written to the observation log (max 1 per person and kind per 30 s); per-event progress bars. |
| `scripts/extract_detections.py` | `--object-detector`, `--det-skip`. |
| `docs/annotation_guide.md`, `docs/event_definitions.md` | Categories `looking_down`, `material_exchange`; group-session seats and lookalikes. |
| `tests/test_week5.py` | Tests: CLIP decision (calculator not a phone), calculator in hand gives no phone event (context: logged), writing pose not looking_down / lap look is, exchange between two seats alerts after 1 s, exchange with invigilator suppressed in context mode. |

Checks run by the AI on Test_1 (author to verify): YOLO-World alone labels calculators "cell phone"
(0.61-0.66); CLIP on crops: calculators 0.98-1.00, B2's phone 0.69, F2's phone 0.75. On a 30 s slice
(55-85 s) world+CLIP found only the two real phones and recognised all four calculators, where the
COCO detector produced 447 frames of "phone in hand" at F2. Normal writing (60-90 s): no
looking_down or material_exchange signal. Speed: 0.33 s per processed frame (~22 min whole video).
