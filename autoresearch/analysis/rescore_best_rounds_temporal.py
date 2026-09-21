"""Re-score saved best-round tensors with the current spatial+temporal contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HARBOR = ROOT / "model-inference" / "harbor"
sys.path.insert(0, str(HARBOR))

import verify  # noqa: E402
from score import aggregate  # noqa: E402


CAMPAIGNS = {
    "codex-sana-round50": {
        "model": "sana_video2",
        "cases": HERE / "sana50/cases.json",
        "eval": HERE / "sana50/runs/round-50/eval",
        "reference": ROOT / "model-inference/outputs/harbor/sana_video2_gpu_resident_all8/references/all",
    },
    "claude-sana-round44": {
        "model": "sana_video2",
        "cases": HERE / "claude_sana50/cases.json",
        "eval": HERE / "claude_sana50/runs/round-44/eval",
        "reference": ROOT / "model-inference/outputs/harbor/sana_video2_gpu_resident_all8/references/all",
    },
    "codex-h3-round19": {
        "model": "minimax_h3",
        "cases": HERE / "h3_25/cases.json",
        "eval": HERE / "h3_25/runs/round-19/eval",
        "reference": ROOT / "model-inference/outputs/harbor/minimax_h3_dense_no_cache_02_04/references/public",
    },
    "claude-h3-round24": {
        "model": "minimax_h3",
        "cases": HERE / "claude_h3_25/cases.json",
        "eval": HERE / "claude_h3_25/runs/round-24/eval",
        "reference": ROOT / "model-inference/outputs/harbor/minimax_h3_dense_no_cache_02_04/references/public",
    },
}


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run tensor re-scoring on a Slurm GPU node")
    reports = {}
    for name, spec in CAMPAIGNS.items():
        cases = json.loads(spec["cases"].read_text())
        measured = json.loads((spec["eval"] / "measurement/reward.json").read_text())
        reference = json.loads((spec["reference"] / "manifest.json").read_text())
        args = SimpleNamespace(
            model=spec["model"], reference=spec["reference"], output=spec["eval"], repeats=1
        )
        rows = verify.score_outputs(args, cases, measured, reference)
        gate = verify.apply_split_quality_gate(spec["model"], rows)
        raw_reward = aggregate(rows)
        previous = json.loads((spec["eval"] / "reward.json").read_text())
        reports[name] = {
            "model": spec["model"],
            "previous_reward_without_temporal_factor": previous.get("reward"),
            "raw_reward_with_temporal_factor": raw_reward,
            "reward_with_temporal_gate": raw_reward if gate["passed"] else 0.0,
            "quality_gate": gate,
            "cases": rows,
        }
    output = HERE / "temporal-tensor-rescore-20260921.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "contract": {
            "case_reward": "speedup * max(0,1-LPIPS) * max(0,1-Excess-tLPx100)",
            "sana_video2_mean_excess_tlp_x100_max": 0.30,
            "minimax_h3_mean_excess_tlp_x100_max": 0.10,
        },
        "campaigns": reports,
    }, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({
        name: {
            "mean_excess_tlp_x100": report["quality_gate"]["mean_excess_tlp_x100"],
            "temporal_passed": report["quality_gate"]["temporal_passed"],
            "raw_reward": report["raw_reward_with_temporal_factor"],
            "final_reward": report["reward_with_temporal_gate"],
        }
        for name, report in reports.items()
    }, indent=2))
    print(output)


if __name__ == "__main__":
    main()
