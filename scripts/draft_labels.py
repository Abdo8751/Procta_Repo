"""First draft of group ground truth from a timed protocol, aligned to the video by the hand raise.

  python scripts/draft_labels.py --protocol docs/protocol_v3_group.csv --video-sync 12.4 \
      --out data/labels/Test_1_gt.csv

Protocol CSV columns: start_s,end_s,event_type,seat,instruction   (seat may be empty)
  * event_type "none" rows are skipped; "a+b" gives one label per event (like record_session.py)
  * the row with event_type "sync" (everyone raises a hand) marks protocol time T_sync;
    --video-sync is the video second where the hands go up. Every label is shifted by
    video_sync - T_sync. Without a sync row, T_sync = 0 (protocol starts at --video-sync).
The draft is then corrected on video with:
  python scripts/label_video.py --video data/videos/Test_1.mp4 --point T1
"""

import argparse
import csv
from pathlib import Path


def draft(protocol_rows, video_sync, protocol_sync=None):
    if protocol_sync is None:
        protocol_sync = next((float(r["start_s"]) for r in protocol_rows if r["event_type"].strip() == "sync"), 0.0)
    shift = video_sync - protocol_sync
    out = []
    for r in protocol_rows:
        ev = r["event_type"].strip()
        if ev in ("none", "sync", ""):
            continue
        for e in ev.split("+"):
            out.append({"event_type": e.strip(), "start_s": round(float(r["start_s"]) + shift, 2),
                        "end_s": round(float(r["end_s"]) + shift, 2), "seat": (r.get("seat") or "").strip(),
                        "note": f"draft: {r.get('instruction', '').strip()}"})
    return sorted(out, key=lambda x: (x["start_s"], x["seat"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--video-sync", type=float, required=True, help="video second where everyone raises a hand")
    ap.add_argument("--protocol-sync", type=float, default=None, help="protocol second of the hand raise "
                    "(default: start of the 'sync' row, else 0)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.protocol, encoding="utf-8")))
    labels = draft(rows, args.video_sync, args.protocol_sync)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists - not overwriting corrected labels. Use another --out.")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["event_type", "start_s", "end_s", "seat", "note"])
        w.writeheader()
        w.writerows(labels)
    print(f"{len(labels)} draft labels -> {args.out}")


if __name__ == "__main__":
    main()
