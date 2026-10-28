"""Live PROCTA: the camera stands in for the robot's camera on a predefined observation route.

  python scripts/live.py --mode context                 # proposed method: temporal + consolidation + exam context
  python scripts/live.py --mode improved                # temporal confirmation + consolidation, no context
  python scripts/live.py --mode frame                   # frame-level baseline (B0)
  python scripts/live.py --mode baseline --record       # 3-s rule (B1), also save the raw video
  python scripts/live.py --mode context --video data/videos/s04.mp4   # replay a recording as if live
  python scripts/live.py --mode context --video data/videos/Test_1.mp4 --start-point T1 \
      --detections data/detections/Test_1_world.jsonl   # smooth demo: replay the saved detections

Robot route (configs/exam_context.json -> observation_points). The operator plays the robot:
  n / p   move to the next / previous permitted observation point (detection re-calibrates)
  --auto-route   after each point's dwell time the screen asks you to move the camera on
If the participant cannot be seen clearly for insufficient_view_s, the system logs an
"insufficient view" observation and asks for the next permitted point - it never alerts on it.

Keys: n/p route, k pose on/off, q quit.
Video replay (--video) also: space pause/resume, j / l back / forward 10 s, , / . back / forward 2 s,
+ / - faster / slower (analyses every Nth frame; video time is kept). A jump resets the running
timers, so behaviour seen again after a rewind is merged into its existing event, not duplicated.
Every confirmed event is logged to results/alerts.db
with timestamp, observation point, image region, a still, and (temporal modes) an evidence clip.
View and review them with:  python scripts/dashboard.py
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from procta.alert_log import AlertLog  # noqa: E402
from procta.camera import open_camera  # noqa: E402
from procta.clips import ClipRecorder  # noqa: E402
from procta.config import Config  # noqa: E402
from procta.context import DEFAULT_PATH, ExamContext  # noqa: E402
from procta.detect import Detector  # noqa: E402
from procta.draw import annotate  # noqa: E402
from procta.events import EventConsolidator  # noqa: E402
from procta.signals import SignalExtractor  # noqa: E402
from procta.temporal import MODES, BaselinePersistence, ConfidenceAwareAlerter, make_policy, signal_mode  # noqa: E402

FACE_OR_HANDS = (0, 1, 2, 9, 10)          # nose, eyes, wrists


def clear_view(dets):
    """At least one person whose face or a wrist is confidently visible."""
    return any(any(p["kps"][i][2] > 0.3 for i in FACE_OR_HANDS) for p in dets["persons"])


def region_of(dets, ident):
    p = next((p for p in dets["persons"] if p.get("key", p["id"]) == ident), None)
    return [round(v) for v in p["box"]] if p else None


def draw_temporal(img, pol, t, cfg, x0=None):
    """The temporal decision, live: one bar per open episode towards the confirmation threshold."""
    rows = []
    if isinstance(pol, ConfidenceAwareAlerter):
        for (tid, ev), ep in pol.eps.items():
            dur = t - ep.start
            ratio = ep.on_time / dur if dur > 0 else 1.0
            p = cfg.persistence_for(ev)
            rows.append((f"{ev} #{tid}", min(dur / p, 1.0), ep.alerted, f"{dur:.1f}/{p:.0f}s on {ratio:.0%}"))
    elif isinstance(pol, BaselinePersistence):
        for (tid, ev), st in pol.start.items():
            dur = t - st
            p = cfg.persistence_for(ev)
            rows.append((f"{ev} #{tid}", min(dur / p, 1.0), (tid, ev) in pol.alerted, f"{dur:.1f}/{p:.0f}s"))
    h, w = img.shape[:2]
    x0 = x0 if x0 is not None else w - 330
    rows = rows[:6]
    y = h - 34 - 18 * len(rows)                 # bottom-right, clear of the alert banners
    for label, frac, confirmed, txt in rows:
        cv2.rectangle(img, (x0, y - 12), (x0 + 120, y), (80, 80, 80), 1)
        cv2.rectangle(img, (x0, y - 12), (x0 + int(120 * frac), y), (0, 0, 255) if confirmed else (0, 200, 255), -1)
        cv2.putText(img, f"{label} {txt}{' CONFIRMED' if confirmed else ''}", (x0 + 126, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mode", choices=list(MODES), default="context")
    ap.add_argument("--context", default=str(DEFAULT_PATH), help="exam context / route file")
    ap.add_argument("--start-point", default=None)
    ap.add_argument("--auto-route", action="store_true", help="prompt to move on after each point's dwell time")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--det-skip", type=int, default=2)
    ap.add_argument("--det-imgsz", type=int, default=0, help="object detector input size (0 = --imgsz)")
    ap.add_argument("--object-detector", choices=["coco", "world"], default="coco",
                    help="world = YOLO-World + CLIP: tells phones from calculators (slower)")
    ap.add_argument("--sync-objects", action="store_true",
                    help="run the object detector in the main loop (default: background thread, lower latency)")
    ap.add_argument("--stride", type=int, default=1, help="--video only: process every Nth frame (video time kept)")
    ap.add_argument("--video", default=None, help="replay a video file as if it were the live camera")
    ap.add_argument("--detections", default=None,
                    help="--video only: use detections saved by extract_detections.py instead of running the "
                         "networks (smooth, real-time replay; identical input to the offline evaluation)")
    ap.add_argument("--headless", action="store_true", help="no window (tests / batch replay)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    cfg = Config(imgsz=args.imgsz, det_imgsz=args.det_imgsz, det_frame_skip=args.det_skip,
                 object_detector=args.object_detector, world_model=str(ROOT / "models/yolov8s-worldv2.pt"),
                 async_objects=not args.sync_objects,
                 det_model=str(ROOT / "models/yolov8n.pt"), pose_model=str(ROOT / "models/yolov8n-pose.pt"))
    ctx = ExamContext.load(args.context)
    # robot route = points marked on_route; a fixed viewpoint (e.g. a recorded room with seat
    # zones) is only used when named with --start-point
    route = [p for p in ctx.observation_points if p.on_route or p.id == args.start_point] or ctx.observation_points
    pi = next((i for i, p in enumerate(route) if p.id == args.start_point), 0)

    started_at = datetime.now()
    name = f"live_{started_at:%Y%m%d_%H%M%S}_{args.mode}"
    out = ROOT / "results" / name
    (out / "evidence").mkdir(parents=True, exist_ok=True)

    if args.video:
        cap = cv2.VideoCapture(args.video)
        vfps = cap.get(cv2.CAP_PROP_FPS) or 15
    else:
        cap = open_camera(args.camera, args.width, args.height)
    show_pose = True
    ok, first = cap.read()
    alert_log = AlertLog(ROOT / "results/alerts.db", name, args.mode, started_at, args.camera,
                         f"{first.shape[1]}x{first.shape[0]}" if ok else "",
                         {**cfg.to_dict(), "exam_context": ctx.to_dict() if args.mode == "context" else None})
    cached = None
    if args.detections:                      # replay saved detections: no neural networks in the loop
        import bisect
        from procta.detect import load_detections
        _, cframes = load_detections(Path(args.detections))
        cached = {f["frame"]: f for f in cframes}
        cached_idx = sorted(cached)
        det = None
    else:
        det = Detector(cfg)
    sig = SignalExtractor(cfg, signal_mode(args.mode), context=ctx, point=route[pi])
    pol = make_policy(args.mode, cfg)
    temporal = args.mode in ("improved", "context")
    cons = EventConsolidator(ctx.consolidation_window_s) if temporal else None
    clips = ClipRecorder(out / "evidence") if temporal else None
    rel = lambda p: Path(p).relative_to(ROOT / "results").as_posix()  # noqa: E731

    writer, times = None, []
    log = open(out / "alerts.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["track_id", "event_type", "t_alert", "t_start", "confidence", "positive_ratio", "evidence",
                 "observation_point", "event_id", "sighting"])
    recent = []                         # (until_t, text) banners
    current_clip = {}                   # event id -> clip name being recorded
    start, prev = time.time(), time.time()
    alert_log.started_at = datetime.fromtimestamp(start)
    arrived, last_clear, insufficient_flagged = 0.0, 0.0, False
    alert_log.observe(0.0, route[pi].id, "arrive", route[pi].name)
    print(f"Mode={args.mode}. Point {route[pi].id} ({route[pi].name}). Sit still ~2 s for calibration.")
    print("Keys: n/p next/previous observation point, k pose, q quit.  Dashboard: python scripts/dashboard.py")

    def end_episodes(closed, t):
        for key, st, last_on, alerted in closed:
            if not (alerted and cons):
                continue
            ev = cons.on_episode_end(key, last_on, route[pi].id)
            if ev is None:
                continue
            if ev.id in current_clip:
                clips.stop(current_clip.pop(ev.id), last_on)
            if ev.log_id:
                alert_log.update_event(ev.log_id, ev.sightings, ev.t_last_seen)

    def move_to(new_i, t):
        nonlocal pi, arrived, last_clear, insufficient_flagged
        end_episodes(pol.flush() if hasattr(pol, "flush") else [], t)
        alert_log.observe(t, route[pi].id, "leave", f"to {route[new_i].id}")
        pi = new_i
        sig.set_point(route[pi])
        arrived, last_clear, insufficient_flagged = t, t, False
        alert_log.observe(t, route[pi].id, "arrive", route[pi].name)
        recent.append((t + 3, f"Robot at point {route[pi].id}: {route[pi].name} - calibrating"))
        print(f"[{t:7.1f}s] moved to point {route[pi].id} ({route[pi].name})")

    n_frame = 0
    last_note = {}                      # (identity, kind) -> time of the last logged context note
    stride, paused, last_img = max(1, args.stride), False, None
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.video else 0
    speed = 1.0                         # cached replay: playback speed (real time = 1)
    clock = {"wall": time.time(), "video": 0.0}     # pacing reference for cached replay

    def reset_clock(t_video):
        clock["wall"], clock["video"] = time.time(), t_video

    def seek(seconds):
        """Video replay: jump in time. Running episodes end and per-person motion history is
        cleared, because the timers cannot run backwards; events already logged stay."""
        nonlocal n_frame, pol, recent, last_img
        t_now = n_frame / vfps
        target = int(min(max(n_frame + seconds * vfps, 0), max(n_total - 1, 0)))
        if temporal:
            end_episodes(pol.flush(), t_now)
            clips.stop_all(t_now)
            clips.buffer.clear()
            clips._last_t = None
        pol = make_policy(args.mode, cfg)
        sig.prev_shoulder.clear()
        recent = [(target / vfps + 2, f"jumped to {target / vfps:.0f}s")]
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        n_frame = target
        last_img = None                         # paused: show the frame at the new position once
        reset_clock(target / vfps)

    def handle_key(key, t):
        """-> True to quit."""
        nonlocal show_pose, paused, stride, speed
        if key == ord("q"):
            return True
        if key == ord("k"):
            show_pose = not show_pose
        if key in (ord("n"), ord("p")) and len(route) > 1:
            move_to((pi + (1 if key == ord("n") else -1)) % len(route), t)
        if args.video:
            if key == ord(" "):
                paused = not paused
                reset_clock(n_frame / vfps)
            jump = {ord("j"): -10, ord("l"): 10, ord(","): -2, ord("."): 2}.get(key)
            if jump:
                seek(jump)
            if key in (ord("+"), ord("="), ord("-"), ord("_")):
                up = key in (ord("+"), ord("="))
                if cached is not None:               # cached replay: real playback speed
                    speed = min(speed * 2, 16.0) if up else max(speed / 2, 0.25)
                    reset_clock(n_frame / vfps)
                else:                                # networks in the loop: analyse every Nth frame
                    stride = min(stride + 1, 15) if up else max(stride - 1, 1)
        return False

    try:
        while True:
            if paused and last_img is not None:
                view = last_img.copy()
                cv2.putText(view, "PAUSED - space resume, j/l -/+10 s, ,/. -/+2 s, q quit", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("PROCTA live", view)
                if handle_key(cv2.waitKey(50) & 0xFF, n_frame / vfps):
                    break
                continue
            if cached is not None:              # only frames with saved detections are shown
                i = bisect.bisect_left(cached_idx, n_frame)
                nxt = cached_idx[i] if i < len(cached_idx) else n_total
                while n_frame < nxt and cap.grab():
                    n_frame += 1
            elif args.video and stride > 1 and n_frame:
                for _ in range(stride - 1):     # skipped frames: decode only, video time advances
                    if cap.grab():
                        n_frame += 1
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            t = n_frame / vfps if args.video else now - start      # video replay uses video time
            n_frame += 1
            fps, prev = 1.0 / max(now - prev, 1e-6), now
            if args.record:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
                writer.write(frame)
                times.append(t)

            point = route[pi]
            if cached is not None:
                c = cached.get(n_frame - 1)
                dets = {**c, "t": t, "persons": [dict(p) for p in c["persons"]]} if c else \
                    {"t": t, "wh": [frame.shape[1], frame.shape[0]], "persons": [], "phones": [], "objects": []}
            else:
                dets = det(frame, t)
            signals = sig(dets)
            alerts = pol.step(t, signals, sig.present)
            img = annotate(frame, dets, signals, show_pose=show_pose) if alerts else None   # evidence still

            for a in alerts:
                who = a.track_id if isinstance(a.track_id, str) else f"track{a.track_id}"
                ev_file = out / "evidence" / f"{a.t_alert:07.1f}s_{point.id}_{who}_{a.event_type}.jpg"
                cv2.imwrite(str(ev_file), img)
                region = region_of(dets, a.track_id)
                eid, sighting = "", 1
                if cons is None:
                    alert_log.add(a, rel(ev_file), point.id, region)
                else:
                    ev, is_new = cons.on_alert(a, point.id)
                    eid, sighting = ev.id, ev.sightings
                    clip_name = f"ev{ev.id:03d}_s{ev.sightings}_{a.event_type}"
                    clip_file = clips.start(clip_name, a.t_start)
                    current_clip[ev.id] = clip_name
                    if is_new:
                        note = (f"neighbour side: {point.neighbour_side or '-'}; allowed look: "
                                f"{', '.join(point.allowed_look_sides) or '-'} ({point.allowed_look_reason})"
                                if args.mode == "context" else None)
                        ev.log_id = alert_log.add(a, rel(ev_file), point.id, region,
                                                  rel(out / "evidence" / clip_file), note)
                    else:
                        alert_log.update_event(ev.log_id, ev.sightings, ev.t_last_seen,
                                               rel(out / "evidence" / clip_file))
                wr.writerow([a.track_id, a.event_type, f"{a.t_alert:.2f}", f"{a.t_start:.2f}", a.confidence,
                             a.positive_ratio, ev_file.name, point.id, eid, sighting])
                log.flush()
                conf = "" if a.confidence != a.confidence else f" (confidence {a.confidence:.2f})"
                if sighting > 1:
                    msg = f"repeated sighting #{sighting} added to event {eid} ({a.event_type})"
                else:
                    msg = a.message(datetime.now().strftime("%H:%M:%S"))
                recent.append((t + 4, f"[{point.id}] {msg}{conf}"))
                print(f"[{t:7.1f}s] [{point.id}] {'seat' if isinstance(a.track_id, str) else 'track'} {a.track_id}: {msg}{conf}")

            if temporal:
                end_episodes(pol.closed, t)

            # --- context notes: logged in the observation log, never alerted (one per person per 30 s)
            for nt, who, kind, text in sig.notes:
                if nt - last_note.get((who, kind), -1e9) >= 30:
                    last_note[(who, kind)] = nt
                    alert_log.observe(nt, point.id, kind, f"{who}: {text}")
                    print(f"[{nt:7.1f}s] [{point.id}] note {who}: {text}")
            sig.notes.clear()

            # --- view quality at this observation point
            if clear_view(dets):
                last_clear, insufficient_flagged = t, False
            elif t - last_clear > ctx.insufficient_view_s and t - arrived > ctx.insufficient_view_s and not insufficient_flagged:
                nxt = route[(pi + 1) % len(route)]
                alert_log.observe(t, point.id, "insufficient_view",
                                  f"no clear view for {ctx.insufficient_view_s:.0f}s; next permitted point {nxt.id}")
                recent.append((t + 6, f"Insufficient view at {point.id} - move to permitted point {nxt.id} (press n)"))
                print(f"[{t:7.1f}s] insufficient view at {point.id}; suggest point {nxt.id}")
                insufficient_flagged = True
            if args.auto_route and len(route) > 1 and t - arrived > point.dwell_s:
                nxt = route[(pi + 1) % len(route)]
                cv2.putText(img, f"ROUTE: move camera to point {nxt.id} ({nxt.name}) and press n", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            recent = [r for r in recent if r[0] > t]
            img = annotate(frame, dets, signals, [r[1] for r in recent], show_pose=show_pose,
                           seat_zones=point.seat_zones)
            for p in dets["persons"]:             # head yaw / pitch vs calibration, to tune the thresholds
                who = p.get("key", p["id"])
                yaw, pitch = sig.head_yaw.get(who), sig.head_pitch.get(who)
                if yaw is not None:
                    turned = abs(yaw) > sig.cfg.head_yaw_thr
                    cv2.putText(img, f"yaw {yaw:+.2f} ({'image right' if yaw > 0 else 'image left'})",
                                (int(p["box"][0]), int(p["box"][1]) + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if turned else (0, 200, 0), 1)
                if pitch is not None:
                    down = (who, "looking_down") in signals
                    cv2.putText(img, f"pitch {pitch:+.2f} ({'hands below desk' if down else 'writing pose'})",
                                (int(p["box"][0]), int(p["box"][1]) + 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if down else (0, 200, 0), 1)
            draw_temporal(img, pol, t, cfg)
            if clips is not None:
                clips.push(t, img)
            if not dets["persons"]:
                cv2.putText(img, "no person detected - sit further back / more light", (10, img.shape[0] - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            supp = sum(sig.suppressed.values())
            spd = f"speed {speed:g}x (saved detections)" if cached is not None else f"every {stride}. frame"
            vid = (f"  {int(t) // 60}:{int(t) % 60:02d}/{int(n_total / vfps) // 60}:{int(n_total / vfps) % 60:02d}"
                   f"  {spd}  [space pause  j/l -/+10s  +/- speed]") if args.video else ""
            cv2.putText(img, f"FPS {fps:.1f}  t={t:.0f}s  mode={args.mode}  point {point.id}"
                             f"{f'  context-suppressed {supp}' if args.mode == 'context' else ''}{vid}",
                        (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            last_img = img
            if args.headless:
                key = 255
            else:
                cv2.imshow("PROCTA live", img)
                wait = 1
                if cached is not None:          # pace the replay to (speed x) real time
                    wait = max(1, int(((clock["wall"] + (t - clock["video"]) / speed) - time.time()) * 1000))
                key = cv2.waitKey(min(wait, 500)) & 0xFF
            if handle_key(key, t):
                break
    finally:
        t_end = n_frame / vfps if args.video else time.time() - start
        if temporal:
            end_episodes(pol.flush(), t_end)
            clips.stop_all(t_end)
            clips.join()
        alert_log.observe(t_end, route[pi].id, "leave", "session end")
        if args.mode == "context" and sig.suppressed:
            print("Context suppressed:", dict(sig.suppressed))
        cap.release()
        log.close()
        alert_log.close()
        if writer is not None:
            writer.release()
            with open(out / f"{name}_times.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "t"])
                w.writerows([i, f"{x:.4f}"] for i, x in enumerate(times))
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
