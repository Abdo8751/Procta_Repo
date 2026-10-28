"""Temporal alerting policies: per-frame signals -> discrete, reviewable alerts.

FrameLevelAlerter    - B0, simple frame-level baseline: alert as soon as a signal is on.

BaselinePersistence  - B1, the rule from procta_live_baseline.py: alert once a signal
                       has been on for persistence_s *without a single off frame*.
                       One missed detection resets the timer, and after the episode
                       "ends" the next on-frame starts a fresh one -> a single long
                       phone use is split into several alerts (fragmentation), and a
                       flickery true episode may never alert at all.

ConfidenceAwareAlerter - the proposed improvement:
  * episodes survive detector drop-outs shorter than gap_tolerance_s
  * an episode alerts once it is >= persistence_s long AND the signal was actually
    on for >= min_positive_ratio of it
  * exactly one alert per episode (less human review effort)
  * each alert carries a confidence = mean detector confidence x positive ratio,
    so the dashboard can rank / triage alerts instead of showing a flat list
"""

from dataclasses import dataclass, asdict

from .config import Config, EVENT_TYPES


@dataclass
class Alert:
    track_id: int           # identity: tracker id, or the seat label (str) when seats are defined
    event_type: str
    t_alert: float          # time the alert fired (video seconds)
    t_start: float          # time the underlying episode started
    confidence: float = float("nan")
    positive_ratio: float = float("nan")

    def to_row(self):
        return asdict(self)

    def message(self, clock: str = None) -> str:
        """Review prompt wording - never a verdict about the student."""
        what = {"phone_in_hand": "mobile-phone interaction", "phone_on_desk": "mobile phone on desk",
                "body_movement": "sustained body movement",
                "unusual_seat_movement": "unusual seat/desk movement",
                "head_turned": "sustained head turn"}.get(self.event_type, self.event_type)
        when = clock or f"{self.t_alert:.1f}s"
        return f"Potential {what} detected at {when} - please review."


class BaselinePersistence:
    def __init__(self, cfg: Config):
        self.thr = cfg.persistence_s
        self.start, self.alerted = {}, set()

    def step(self, t: float, signals: dict, present_tracks: set):
        alerts = []
        # baseline only updates the 4 events of tracks visible in this frame
        for tid in present_tracks:
            for ev in EVENT_TYPES:
                key = (tid, ev)
                if key not in signals:
                    self.start.pop(key, None)
                    self.alerted.discard(key)
                    continue
                self.start.setdefault(key, t)
                if t - self.start[key] >= self.thr and key not in self.alerted:
                    self.alerted.add(key)
                    alerts.append(Alert(tid, ev, t, self.start[key], signals[key], 1.0))
        return alerts


class _Episode:
    __slots__ = ("start", "last_on", "last_t", "on_time", "conf_sum", "n_on", "alerted")

    def __init__(self, t):
        self.start = self.last_on = self.last_t = t
        self.on_time = 0.0
        self.conf_sum = 0.0
        self.n_on = 0
        self.alerted = False


class ConfidenceAwareAlerter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.eps = {}
        self.closed = []        # episodes that ended in the last step: (key, start, last_on, alerted)

    def flush(self):
        """End every open episode (e.g. the robot leaves this observation point)."""
        self.closed = [(k, ep.start, ep.last_on, ep.alerted) for k, ep in self.eps.items()]
        self.eps = {}
        return self.closed

    def last_seen(self, key):
        ep = self.eps.get(key)
        return ep.last_on if ep else None

    def step(self, t: float, signals: dict, present_tracks: set = None):
        cfg, alerts = self.cfg, []
        self.closed = []
        for key, conf in signals.items():
            ep = self.eps.get(key)
            if ep is None:
                ep = self.eps[key] = _Episode(t)
            else:
                # time since last frame counts as "on" only if previous frame was on
                if ep.last_on == ep.last_t:
                    ep.on_time += t - ep.last_t
                ep.last_on = t
            ep.last_t = t
            ep.conf_sum += conf
            ep.n_on += 1

        for key in list(self.eps):
            ep = self.eps[key]
            if key not in signals:
                ep.last_t = t                  # off time is not credited to on_time
                if t - ep.last_on > cfg.gap_tolerance_s:
                    self.closed.append((key, ep.start, ep.last_on, ep.alerted))
                    del self.eps[key]          # episode over
                continue
            dur = t - ep.start
            ratio = ep.on_time / dur if dur > 0 else 1.0
            if not ep.alerted and dur >= cfg.persistence_s and ratio >= cfg.min_positive_ratio:
                conf = (ep.conf_sum / ep.n_on) * min(1.0, ratio)
                ep.alerted = True
                if conf >= cfg.min_alert_confidence:
                    alerts.append(Alert(key[0], key[1], t, ep.start, round(conf, 3), round(ratio, 3)))
        return alerts


class FrameLevelAlerter:
    """Supervisor's simple baseline ("phone detected -> alert"): no time threshold.
    Fires whenever a signal switches on, i.e. one alert per contiguous run of
    positive frames, so every flicker of the detector becomes a new alert."""

    def __init__(self, cfg: Config):
        self.on = set()

    def step(self, t: float, signals: dict, present_tracks: set = None):
        alerts = [Alert(k[0], k[1], t, t, c, 1.0) for k, c in signals.items() if k not in self.on]
        self.on = set(signals)
        return alerts


# B0 frame-level, B1 3-s persistence, temporal confirmation + consolidation,
# temporal confirmation + consolidation + examination context (the focused improvement)
MODES = ("frame", "baseline", "improved", "context")


def make_policy(mode: str, cfg: Config):
    return {"frame": FrameLevelAlerter, "baseline": BaselinePersistence,
            "improved": ConfidenceAwareAlerter, "context": ConfidenceAwareAlerter}[mode](cfg)


def signal_mode(mode: str) -> str:
    return {"frame": "baseline", "baseline": "baseline", "improved": "improved", "context": "context"}[mode]


def run_policy(frames, mode: str, cfg: Config, context=None, point=None, return_events=False):
    """Replay cached detections through signals + temporal policy (+ consolidation for the
    temporal modes) -> list[Alert] (one per reviewable record). return_events=True also
    returns the consolidated events and the signal extractor (for suppression counts)."""
    from .events import EventConsolidator
    from .signals import SignalExtractor
    sig = SignalExtractor(cfg, signal_mode(mode), context=context, point=point)
    pol = make_policy(mode, cfg)
    window = context.consolidation_window_s if context is not None else 30.0
    cons = EventConsolidator(window) if mode in ("improved", "context") else None
    pid = sig.point.id if sig.point is not None else "A"
    alerts = []
    for fr in frames:
        s = sig(fr)
        new = pol.step(fr["t"], s, sig.present)
        if cons is None:
            alerts += new
            continue
        for a in new:
            ev, is_new = cons.on_alert(a, pid)
            if is_new:
                alerts.append(a)
        for key, start, last_on, alerted in pol.closed:
            if alerted:
                cons.on_episode_end(key, last_on, pid)
    if return_events:
        return alerts, (cons.events if cons else []), sig
    return alerts
