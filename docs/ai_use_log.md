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
