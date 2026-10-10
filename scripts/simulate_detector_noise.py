"""Signal-level simulation: how do the two alerting policies behave when the phone
detector flickers?

Why: on a CPU webcam setup YOLOv8n misses the phone in some frames (motion blur,
hand occlusion, phone seen edge-on). The baseline resets its 3 s timer on ANY missed
frame. This simulation isolates that effect with known ground truth, before the
real staged-video benchmark is labelled. It is NOT a substitute for real video
results - it checks the mechanism and gives an expected effect size.

Model (per frame, 12 fps ~ measured live FPS on a laptop CPU):
  * 10-min session, phone_in_hand episodes of 3-40 s separated by 10-40 s gaps,
    plus short 1-2.5 s "glances" that must NOT alert (hard negatives)
  * inside an episode the detector follows a 2-state (Gilbert-Elliott) chain:
    long-run miss rate = m, mean drop-out length 0.4 s (bursty misses)
  * outside episodes: false-positive bursts ~1.5/min, mean length 1.5 s, low confidence

Usage:  python scripts/simulate_detector_noise.py           (writes results/simulation/)
"""

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from procta.config import Config  # noqa: E402
from procta.evaluation import evaluate  # noqa: E402
from procta.temporal import BaselinePersistence, ConfidenceAwareAlerter, FrameLevelAlerter  # noqa: E402

FPS = 12.0
DUR = 600.0
EV = "phone_in_hand"


def make_session(rng):
    gt, t = [], 10.0
    while t < DUR - 45:
        if rng.random() < 0.25:
            L = rng.uniform(1.0, 2.5)          # glance: should NOT alert
        else:
            L = rng.uniform(3.5, 40.0)
        gt.append({"event_type": EV, "start_s": t, "end_s": t + L})
        t += L + rng.uniform(10, 40)
    return gt


def observe(rng, gt, miss, mean_drop_s=0.4, fp_per_min=1.5, fp_mean_s=1.5):
    p_rec = 1.0 / (mean_drop_s * FPS)
    p_drop = miss * p_rec / (1 - miss) if miss < 1 else 1.0
    p_fp_start = fp_per_min / 60.0 / FPS
    p_fp_end = 1.0 / (fp_mean_s * FPS)
    frames, on, fp_on = [], True, False
    n = int(DUR * FPS)
    for i in range(n):
        t = i / FPS
        inside = any(g["start_s"] <= t <= g["end_s"] for g in gt)
        if inside:
            on = (rng.random() >= p_drop) if on else (rng.random() < p_rec)
            sig = {(1, EV): rng.uniform(0.4, 0.9)} if on else {}
            fp_on = False
        else:
            on = True
            fp_on = (rng.random() >= p_fp_end) if fp_on else (rng.random() < p_fp_start)
            sig = {(1, EV): rng.uniform(0.25, 0.5)} if fp_on else {}
        frames.append((t, sig))
    return frames


def run(policy, frames):
    alerts = []
    for t, sig in frames:
        for a in policy.step(t, sig, {1}):
            alerts.append({"event_type": a.event_type, "t_alert": a.t_alert})
    return alerts


def main(seeds=20):
    out = ROOT / "results/simulation"
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    rows = []
    for miss in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        for seed in range(seeds):
            rng = random.Random(seed * 1000 + int(miss * 100))
            gt = make_session(rng)
            frames = observe(rng, gt, miss)
            for name, pol in (("frame", FrameLevelAlerter(cfg)), ("baseline", BaselinePersistence(cfg)),
                              ("improved", ConfidenceAwareAlerter(cfg))):
                m = evaluate(run(pol, frames), gt, DUR, cfg.persistence_s, 1.0, [EV])[EV]
                rows.append({"miss_rate": miss, "seed": seed, "policy": name, **m})

    with open(out / "sweep_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def mean(vals):
        vals = [v for v in vals if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = []
    for miss in sorted({r["miss_rate"] for r in rows}):
        for pol in ("frame", "baseline", "improved"):
            sel = [r for r in rows if r["miss_rate"] == miss and r["policy"] == pol]
            summary.append({"miss_rate": miss, "policy": pol,
                            **{k: round(mean([r[k] for r in sel]), 3) for k in
                               ("precision", "recall", "false_alarms_per_min", "mean_latency_s",
                                "alerts_per_episode", "alerts")}})

    # ablation of the gap tolerance at a realistic 20 % miss rate
    ablation = []
    for gap in (0.0, 0.25, 0.5, 1.0, 2.0):
        for ratio in (0.6,):
            c = Config(gap_tolerance_s=gap, min_positive_ratio=ratio)
            ms = []
            for seed in range(seeds):
                rng = random.Random(seed * 1000 + 20)
                gt = make_session(rng)
                frames = observe(rng, gt, 0.2)
                ms.append(evaluate(run(ConfidenceAwareAlerter(c), frames), gt, DUR, 3.0, 1.0, [EV])[EV])
            ablation.append({"gap_tolerance_s": gap, "min_positive_ratio": ratio,
                             **{k: round(mean([m[k] for m in ms]), 3) for k in
                                ("precision", "recall", "false_alarms_per_min", "alerts_per_episode")}})

    (out / "summary.json").write_text(json.dumps({"sweep": summary, "gap_ablation_at_miss_0.2": ablation}, indent=1))
    md = ["| miss rate | policy | precision | recall | false alarms/min | latency (s) | alerts/episode |",
          "|---|---|---|---|---|---|---|"]
    for s in summary:
        md.append(f"| {s['miss_rate']:.2f} | {s['policy']} | {s['precision']:.2f} | {s['recall']:.2f} | "
                  f"{s['false_alarms_per_min']:.2f} | {s['mean_latency_s']:.2f} | {s['alerts_per_episode']:.2f} |")
    md += ["", "Gap-tolerance ablation (improved policy, miss rate 0.20):", "",
           "| gap tolerance (s) | precision | recall | false alarms/min | alerts/episode |", "|---|---|---|---|---|"]
    for a in ablation:
        md.append(f"| {a['gap_tolerance_s']} | {a['precision']:.2f} | {a['recall']:.2f} | "
                  f"{a['false_alarms_per_min']:.2f} | {a['alerts_per_episode']:.2f} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        colors = {"frame": "#d4a017", "baseline": "#8a8f98", "improved": "#1f6feb"}
        for ax, key, title in zip(axes, ("recall", "alerts_per_episode", "false_alarms_per_min"),
                                  ("Recall (episodes detected)", "Alerts per true episode (review load)",
                                   "False alarms per minute")):
            for pol in ("frame", "baseline", "improved"):
                xs = [s["miss_rate"] for s in summary if s["policy"] == pol]
                ys = [s[key] for s in summary if s["policy"] == pol]
                ax.plot(xs, ys, marker="o", color=colors[pol], label=pol, linewidth=2)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("per-frame detector miss rate")
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylim(0, 1.05)
        axes[1].set_yscale("log")
        axes[2].set_yscale("log")
        axes[0].legend(frameon=False)
        fig.suptitle("Simulated phone-detector flicker: frame-level vs 3-s persistence vs temporal/contextual alerting "
                     f"({seeds} sessions x 10 min per point)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "sweep.png", dpi=150)
        print(f"plot -> {out / 'sweep.png'}")
    except ImportError:
        print("matplotlib not installed - skipped plot")


if __name__ == "__main__":
    main()
