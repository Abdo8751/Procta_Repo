"""Descriptive analysis of the live-baseline alert log (procta_events_log.csv).

There is no ground truth for these sessions, so this does NOT give precision/recall.
It quantifies two baseline failure patterns visible in the log:
  * re-fires: the same track + event alerting again within REFIRE_S seconds - one
    real episode split into several alerts by a single missed detection
  * simultaneous alerts on different track ids within 1 s - either a tracker ID
    switch / duplicate person, or one phone attributed to two people

Usage: python scripts/analyze_live_log.py --log data/live_log_2026-10.csv
"""

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REFIRE_S = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(Path(__file__).resolve().parents[1] / "data/live_log_2026-10.csv"))
    args = ap.parse_args()
    rows = [{**r, "ts": datetime.fromisoformat(r["timestamp"])} for r in csv.DictReader(open(args.log))]
    rows.sort(key=lambda r: r["ts"])

    sessions, cur = [], [rows[0]]
    for r in rows[1:]:
        if r["ts"] - cur[-1]["ts"] > timedelta(minutes=5):
            sessions.append(cur)
            cur = []
        cur.append(r)
    sessions.append(cur)

    print(f"{len(rows)} alerts in {len(sessions)} sessions\n")
    print("| session start | first-to-last alert (min) | alerts | by type | re-fires <{0}s | multi-track same second |".format(REFIRE_S))
    print("|---|---|---|---|---|---|")
    tot_refire = tot_multi = 0
    gaps = []
    for s in sessions:
        span = (s[-1]["ts"] - s[0]["ts"]).total_seconds() / 60
        by_type = Counter(r["event_type"] for r in s)
        last = {}
        refire = 0
        for r in s:
            k = (r["track_id"], r["event_type"])
            if k in last:
                g = (r["ts"] - last[k]).total_seconds()
                gaps.append(g)
                if g < REFIRE_S:
                    refire += 1
            last[k] = r["ts"]
        per_sec = defaultdict(set)
        for r in s:
            per_sec[r["timestamp"]].add(r["track_id"])
        multi = sum(1 for v in per_sec.values() if len(v) > 1)
        tot_refire += refire
        tot_multi += multi
        types = ", ".join(f"{k}:{v}" for k, v in by_type.items())
        print(f"| {s[0]['timestamp']} | {span:.1f} | {len(s)} | {types} | {refire} | {multi} |")
    print(f"\nTotal re-fires within {REFIRE_S}s: {tot_refire}/{len(rows)} alerts "
          f"({100 * tot_refire / len(rows):.0f}%)")
    print(f"Seconds with alerts on >1 track id: {tot_multi}")
    if gaps:
        gaps.sort()
        print(f"Median gap between repeated alerts of same track+event: {gaps[len(gaps) // 2]:.0f}s "
              f"(min {gaps[0]:.0f}s)")
    tracks = Counter(r["track_id"] for r in rows)
    print(f"Distinct track ids: {dict(tracks)}")


if __name__ == "__main__":
    main()
