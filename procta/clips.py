"""Evidence clips from a rolling video buffer.

The last `pre_s` seconds of (annotated) frames are always kept in memory. When an event
is confirmed - which happens persistence_s AFTER it started - the clip therefore still
begins before the behaviour started. Recording continues until the episode ends (plus
`post_s`), capped at `max_s`. Each clip is written twice, in a background thread:

  <name>.webp  animated, plays directly in the dashboard (every browser supports it)
  <name>.mp4   for VLC / download (mp4v codec; browsers may not play it)
"""

import threading
from collections import deque
from pathlib import Path

import cv2


class ClipRecorder:
    def __init__(self, out_dir, pre_s=6.0, post_s=1.5, max_s=30.0, fps=6.0, width=480):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.pre_s, self.post_s, self.max_s, self.fps, self.width = pre_s, post_s, max_s, fps, width
        self.buffer = deque()
        self.active = {}             # name -> {"frames": [(t, img)], "end": float or None, "t0": float}
        self._last_t = None
        self.threads = []

    def _small(self, frame):
        h, w = frame.shape[:2]
        if w <= self.width:
            return frame.copy()
        return cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)

    def push(self, t, frame):
        """Call once per frame with the annotated frame. Returns names of clips that were finished."""
        if self._last_t is not None and t - self._last_t < 0.9 / self.fps:
            return self._finish_due(t)
        self._last_t = t
        img = self._small(frame)
        self.buffer.append((t, img))
        while self.buffer and t - self.buffer[0][0] > self.pre_s:
            self.buffer.popleft()
        for c in self.active.values():
            c["frames"].append((t, img))
        return self._finish_due(t)

    def start(self, name, t_from):
        """Begin a clip whose first frame is the buffered frame at or after t_from - 1 s."""
        frames = [(t, f) for t, f in self.buffer if t >= t_from - 1.0]
        self.active[name] = {"frames": frames, "end": None, "t0": frames[0][0] if frames else t_from}
        return f"{name}.webp"

    def stop(self, name, t_end):
        if name in self.active and self.active[name]["end"] is None:
            self.active[name]["end"] = t_end + self.post_s

    def stop_all(self, t):
        for name in list(self.active):
            self.stop(name, t)
        return self._finish_due(float("inf"))

    def _finish_due(self, t):
        done = []
        for name, c in list(self.active.items()):
            too_long = c["frames"] and c["frames"][-1][0] - c["t0"] >= self.max_s
            if (c["end"] is not None and t >= c["end"]) or too_long:
                frames = [f for ft, f in c["frames"] if c["end"] is None or ft <= c["end"]]
                del self.active[name]
                th = threading.Thread(target=self._write, args=(name, frames), daemon=True)
                th.start()
                self.threads.append(th)
                done.append(name)
        return done

    def _write(self, name, frames):
        if not frames:
            return
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(self.out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        from PIL import Image
        ims = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        try:
            ims[0].save(self.out_dir / f"{name}.webp", save_all=True, append_images=ims[1:],
                        duration=int(1000 / self.fps), loop=0, quality=60)
        except Exception as e:                     # Pillow without WebP: fall back to GIF
            print(f"[clips] webp failed ({e}); writing GIF")
            ims[0].save(self.out_dir / f"{name}.gif", save_all=True, append_images=ims[1:],
                        duration=int(1000 / self.fps), loop=0)

    def join(self, timeout=10):
        for th in self.threads:
            th.join(timeout)
