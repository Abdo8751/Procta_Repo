"""Convert a phone recording (often 4K HEVC .MOV) once into a processable H.264 .mp4.

  python scripts/convert_video.py "C:/Users/Ar/Downloads/Test_1.MOV" data/videos/Test_1.mp4

* H.264 (libx264, CRF 20) - decodes several times faster than HEVC and plays everywhere
* scaled to at most --max-width px wide (default 1920), aspect ratio kept
* rotation metadata is applied (ffmpeg auto-rotates), so the output is upright
* audio is dropped (not used, and it is personal data)

Uses the ffmpeg binary bundled with the imageio-ffmpeg pip package, so no system
ffmpeg install is needed. Output goes to data/videos/, which is git-ignored.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max-width", type=int, default=1920)
    ap.add_argument("--crf", type=int, default=20)
    args = ap.parse_args()
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", args.src,
           "-vf", f"scale='min({args.max_width},iw)':-2", "-c:v", "libx264", "-preset", "medium",
           "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", args.dst]
    print(" ".join(cmd))
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
