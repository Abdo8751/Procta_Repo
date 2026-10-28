"""Event consolidation: repeated sightings of the same behaviour become ONE reviewable record.

The temporal alerter turns flickering frames into episodes. But a student can also put
the phone away and take it out again 10 s later, or the robot can see the same head turn
again after a short occlusion. Without consolidation each of those is a new alert the
invigilator has to open. Here a new confirmed episode is attached to an existing event
when it has the same event type, is seen from the same observation point, and starts
within `window_s` of the event's last sighting. The evidence of every sighting is kept.

Track ids are not used for matching: the pose tracker often gives the same student a new
id after an occlusion (4 ids for one person in the live log), so matching on id would
split one event into several. Seats ARE stable, so when identities are seat labels
(several students in view) an event only absorbs sightings from the same seat.
"""

from dataclasses import dataclass, field


@dataclass
class Event:
    id: int
    event_type: str
    point: str
    t_start: float               # start of the first sighting (video / session seconds)
    t_alert: float               # when the first sighting was confirmed
    t_last_seen: float
    confidence: float
    track_ids: set = field(default_factory=set)
    sightings: int = 1
    sighting_starts: list = field(default_factory=list)
    clips: list = field(default_factory=list)
    log_id: int = None           # row id in the alert log, if logged


class EventConsolidator:
    def __init__(self, window_s: float = 30.0):
        self.window_s = window_s
        self.events = []

    @staticmethod
    def _same_person(ev, ident):
        seats = {i for i in ev.track_ids if isinstance(i, str)}
        return not (seats or isinstance(ident, str)) or ident in seats

    def _match(self, event_type, point, t_start, ident=None):
        for ev in reversed(self.events):
            if (ev.event_type == event_type and ev.point == point and t_start - ev.t_last_seen <= self.window_s
                    and self._same_person(ev, ident)):
                return ev
        return None

    def on_alert(self, alert, point="A"):
        """alert: temporal.Alert. Returns (event, is_new)."""
        conf = 0.0 if alert.confidence != alert.confidence else float(alert.confidence)
        ev = self._match(alert.event_type, point, alert.t_start, alert.track_id)
        if ev is not None:
            ev.sightings += 1
            ev.sighting_starts.append(alert.t_start)
            ev.t_last_seen = max(ev.t_last_seen, alert.t_alert)
            ev.confidence = max(ev.confidence, conf)
            ev.track_ids.add(alert.track_id)
            return ev, False
        ev = Event(len(self.events) + 1, alert.event_type, point, alert.t_start, alert.t_alert, alert.t_alert,
                   conf, {alert.track_id}, 1, [alert.t_start])
        self.events.append(ev)
        return ev, True

    def on_episode_end(self, key, last_on, point="A"):
        """An alerted episode ended: extend the matching event's last sighting."""
        ident, event_type = key
        for ev in reversed(self.events):
            if ev.event_type == event_type and ev.point == point and self._same_person(ev, ident):
                ev.t_last_seen = max(ev.t_last_seen, last_on)
                return ev
        return None
