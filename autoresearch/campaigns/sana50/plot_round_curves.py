"""Plot authoritative per-round speed, LPIPS, and gated reward curves."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
STATE = json.loads((HERE / "state.json").read_text())
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

rounds = []
speedups = []
mean_lpips = []
worst_lpips = []
rewards = []
raw_rewards = []
invalid_rounds = []
gate_failed_rounds = []

for row in STATE["history"]:
    number = row["round"]
    result = row.get("result", {})
    cases = result.get("cases") or []
    valid_cases = [case for case in cases if not case.get("invalid", 1)]
    valid = not result.get("invalid", 1) and len(valid_cases) == 8
    gate = result.get("quality_gate") or {}
    rounds.append(number)
    if valid:
        speedups.append(sum(case["speedup"] for case in valid_cases) / 8)
        mean_lpips.append(gate.get("mean_lpips", math.nan))
        worst_lpips.append(gate.get("worst_lpips", math.nan))
        if not gate.get("passed", False):
            gate_failed_rounds.append(number)
    else:
        speedups.append(math.nan)
        mean_lpips.append(math.nan)
        worst_lpips.append(math.nan)
        invalid_rounds.append(number)
    rewards.append(float(result.get("reward", 0.0)))
    raw_rewards.append(float(result.get("raw_reward", result.get("reward", 0.0))))

if rounds != list(range(1, 51)):
    raise ValueError("Expected exactly rounds 1..50")

plt.rcParams.update({
    "figure.figsize": (12, 6.5),
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def finish(path):
    plt.xlim(1, 50)
    plt.xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# 1. Mean speedup across all eight cases.
plt.figure()
plt.plot(rounds, speedups, color="#1769aa", marker="o", markersize=3.5,
         linewidth=1.8, label="Mean speedup (8 cases)")
plt.axhline(1.0, color="#666666", linestyle="--", linewidth=1.2,
            label="GPU-resident baseline (1.0x)")
for index, number in enumerate(gate_failed_rounds):
    value = speedups[number - 1]
    plt.scatter(number, value, s=55, facecolors="none", edgecolors="#e67e22",
                linewidths=1.8, zorder=5,
                label="Quality gate failed" if index == 0 else None)
for index, number in enumerate(invalid_rounds):
    plt.scatter(number, 0.92, marker="x", color="#c62828", s=55,
                linewidths=2, label="Invalid/no measurement" if index == 0 else None)
plt.scatter(50, speedups[-1], marker="*", color="#d81b60", s=180,
            zorder=6, label=f"Winner: {speedups[-1]:.3f}x")
plt.title("SANA Autoresearch: Mean Hot-Inference Speedup by Submission")
plt.xlabel("Submission round")
plt.ylabel("Speedup vs GPU-resident baseline (x)")
plt.ylim(0.85, max(value for value in speedups if math.isfinite(value)) * 1.12)
plt.legend(loc="upper left")
finish(OUT / "01_speedup_by_round.png")


# 2. Split-level LPIPS quality and its two gates.
plt.figure()
plt.plot(rounds, mean_lpips, color="#00897b", marker="o", markersize=3.5,
         linewidth=1.8, label="Mean LPIPS")
plt.plot(rounds, worst_lpips, color="#6a1b9a", marker="o", markersize=3.2,
         linewidth=1.6, label="Worst-case LPIPS")
plt.axhline(0.30, color="#00897b", linestyle="--", linewidth=1.3,
            label="Mean gate = 0.30")
plt.axhline(0.50, color="#6a1b9a", linestyle="--", linewidth=1.3,
            label="Worst gate = 0.50")
for index, number in enumerate(gate_failed_rounds):
    plt.scatter(number, mean_lpips[number - 1], marker="x", color="#ef6c00", s=65,
                linewidths=2, zorder=5,
                label="Quality gate failed" if index == 0 else None)
for index, number in enumerate(invalid_rounds):
    plt.scatter(number, 0.0, marker="x", color="#c62828", s=55,
                linewidths=2, label="Invalid/no LPIPS" if index == 0 else None)
plt.scatter(50, mean_lpips[-1], marker="*", color="#d81b60", s=170, zorder=6)
plt.scatter(50, worst_lpips[-1], marker="*", color="#d81b60", s=170, zorder=6)
plt.title("SANA Autoresearch: LPIPS Quality by Submission")
plt.xlabel("Submission round")
plt.ylabel("LPIPS (lower is better)")
plt.ylim(-0.02, max(0.8, max(value for value in worst_lpips if math.isfinite(value)) * 1.08))
plt.legend(loc="upper left", ncol=2)
finish(OUT / "02_lpips_by_round.png")


# 3. Final gated reward, plus running best for the optimization trajectory.
running_best = []
best = 1.0
for reward in rewards:
    best = max(best, reward)
    running_best.append(best)

plt.figure()
plt.plot(rounds, rewards, color="#2e7d32", marker="o", markersize=3.5,
         linewidth=1.8, label="Final gated reward")
plt.plot(rounds, raw_rewards, color="#90a4ae", linewidth=1.0, alpha=0.75,
         label="Raw reward before global gate")
plt.plot(rounds, running_best, color="#263238", linestyle="--", linewidth=1.5,
         label="Running best")
plt.axhline(1.0, color="#666666", linestyle=":", linewidth=1.2,
            label="GPU-resident baseline score = 1")
for index, number in enumerate(gate_failed_rounds):
    plt.scatter(number, 0.0, marker="x", color="#ef6c00", s=65,
                linewidths=2, zorder=5,
                label="Quality-gated to zero" if index == 0 else None)
for index, number in enumerate(invalid_rounds):
    plt.scatter(number, 0.0, marker="x", color="#c62828", s=55,
                linewidths=2, zorder=5,
                label="Invalid/failed round" if index == 0 else None)
plt.scatter(50, rewards[-1], marker="*", color="#d81b60", s=180,
            zorder=6, label=f"Winner: {rewards[-1]:.3f}")
plt.title("SANA Autoresearch: Gated Composite Score by Submission")
plt.xlabel("Submission round")
plt.ylabel("Reward = mean[speedup × (1 − LPIPS)]")
plt.ylim(-0.08, max(max(raw_rewards), max(rewards)) * 1.12)
plt.legend(loc="upper left", ncol=2)
finish(OUT / "03_composite_score_by_round.png")

print(OUT / "01_speedup_by_round.png")
print(OUT / "02_lpips_by_round.png")
print(OUT / "03_composite_score_by_round.png")
