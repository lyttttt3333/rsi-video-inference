"""Evaluate reference-relative temporal flicker on packaged best-round MP4s."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
CANDIDATE_ROOT = ROOT / "best-round-mp4-snapshot-20260920"
TEACHER_ROOT = ROOT / "teacher-videos-snapshot-20260920"
OUTPUT = ROOT / "temporal-flicker-evaluation-20260920"


def decode_rgb(path: Path, width: int, height: int, expected_frames: int) -> np.ndarray:
    command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    frame_bytes = width * height * 3
    if len(result.stdout) % frame_bytes:
        raise ValueError(f"Invalid decoded byte count for {path}")
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
    if len(frames) != expected_frames:
        raise ValueError(f"{path}: expected {expected_frames} frames, decoded {len(frames)}")
    return frames


def adjacent_rgb_mae(frames: np.ndarray) -> np.ndarray:
    # Match VBench's adjacent-frame RGB MAE, normalized from [0,255] to [0,1].
    values = []
    for index in range(len(frames) - 1):
        a = frames[index].astype(np.float32)
        b = frames[index + 1].astype(np.float32)
        values.append(float(np.mean(np.abs(a - b))) / 255.0)
    return np.asarray(values, dtype=np.float64)


def adjacent_lpips(frames: np.ndarray, metric, torch, batch_size: int = 8) -> np.ndarray:
    values = []
    with torch.inference_mode():
        for start in range(0, len(frames) - 1, batch_size):
            stop = min(start + batch_size, len(frames) - 1)
            before = np.ascontiguousarray(frames[start:stop])
            after = np.ascontiguousarray(frames[start + 1:stop + 1])
            a = torch.from_numpy(before).permute(0, 3, 1, 2).cuda(non_blocking=True)
            b = torch.from_numpy(after).permute(0, 3, 1, 2).cuda(non_blocking=True)
            a = a.float().div_(127.5).sub_(1.0)
            b = b.float().div_(127.5).sub_(1.0)
            values.extend(float(value) for value in metric(a, b).flatten().cpu())
            del a, b
    return np.asarray(values, dtype=np.float64)


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.percentile(values, quantile))


def summaries(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p95": percentile(values, 95),
        "max": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "group", "agent", "model", "round", "case_id", "spatial_lpips",
        "tlp_mean", "tlp_p95", "tlp_max", "excess_tlp_mean",
        "excess_tlp_p95", "excess_tlp_max", "candidate_adjacent_lpips_mean",
        "teacher_adjacent_lpips_mean", "candidate_rgb_mae_mean",
        "teacher_rgb_mae_mean", "vbench_candidate_score", "vbench_teacher_score",
        "vbench_score_delta", "worst_transition_start_frame",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run temporal LPIPS on a Slurm GPU node")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")

    import importlib.metadata
    import torch
    import lpips

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if importlib.metadata.version("lpips") != "0.1.4":
        raise ValueError("LPIPS must be pinned to 0.1.4")
    alex = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    if not alex.is_file():
        raise FileNotFoundError(f"Offline AlexNet weights missing: {alex}")
    metric = lpips.LPIPS(net="alex", version="0.1").eval().cuda()

    candidate_manifest = json.loads((CANDIDATE_ROOT / "manifest.json").read_text())
    teacher_manifest = json.loads((TEACHER_ROOT / "manifest.json").read_text())
    teachers = {
        video["id"]: video
        for model in teacher_manifest["models"].values()
        for video in model["videos"]
    }
    teacher_cache: dict[str, dict[str, object]] = {}
    rows = []
    per_transition = {}
    started = time.monotonic()

    for group_name, group in candidate_manifest["groups"].items():
        agent = group["agent"]
        model = group["model"]
        for case in group["cases"]:
            case_id = case["id"]
            if case_id not in teachers:
                raise KeyError(f"No teacher MP4 for {case_id}")
            teacher_record = teachers[case_id]
            width = int(case["width"])
            height = int(case["height"])
            frames_count = int(case["frames"])

            if case_id not in teacher_cache:
                teacher_frames = decode_rgb(
                    TEACHER_ROOT / teacher_record["mp4"], width, height, frames_count
                )
                teacher_cache[case_id] = {
                    "lpips": adjacent_lpips(teacher_frames, metric, torch),
                    "rgb_mae": adjacent_rgb_mae(teacher_frames),
                }
                del teacher_frames

            candidate_frames = decode_rgb(
                CANDIDATE_ROOT / case["mp4"], width, height, frames_count
            )
            candidate_lpips = adjacent_lpips(candidate_frames, metric, torch)
            candidate_rgb = adjacent_rgb_mae(candidate_frames)
            del candidate_frames

            teacher_lpips = teacher_cache[case_id]["lpips"]
            teacher_rgb = teacher_cache[case_id]["rgb_mae"]
            if candidate_lpips.shape != teacher_lpips.shape:
                raise ValueError(f"Temporal LPIPS length mismatch for {group_name}/{case_id}")
            temporal_error = np.abs(candidate_lpips - teacher_lpips)
            excess_error = np.maximum(candidate_lpips - teacher_lpips, 0.0)
            worst = int(np.argmax(temporal_error))
            t = summaries(temporal_error)
            excess = summaries(excess_error)
            row = {
                "group": group_name,
                "agent": agent,
                "model": model,
                "round": int(group["round"]),
                "case_id": case_id,
                "spatial_lpips": float(case["lpips"]),
                "tlp_mean": t["mean"],
                "tlp_p95": t["p95"],
                "tlp_max": t["max"],
                "excess_tlp_mean": excess["mean"],
                "excess_tlp_p95": excess["p95"],
                "excess_tlp_max": excess["max"],
                "candidate_adjacent_lpips_mean": float(np.mean(candidate_lpips)),
                "teacher_adjacent_lpips_mean": float(np.mean(teacher_lpips)),
                "candidate_rgb_mae_mean": float(np.mean(candidate_rgb)),
                "teacher_rgb_mae_mean": float(np.mean(teacher_rgb)),
                "vbench_candidate_score": 1.0 - float(np.mean(candidate_rgb)),
                "vbench_teacher_score": 1.0 - float(np.mean(teacher_rgb)),
                "vbench_score_delta": float(np.mean(teacher_rgb) - np.mean(candidate_rgb)),
                "worst_transition_start_frame": worst,
            }
            rows.append(row)
            per_transition[f"{group_name}/{case_id}"] = {
                "frame_index_semantics": "entry i compares decoded frames i and i+1; indices are zero-based",
                "teacher_adjacent_lpips": teacher_lpips.tolist(),
                "candidate_adjacent_lpips": candidate_lpips.tolist(),
                "tlp": temporal_error.tolist(),
                "excess_tlp": excess_error.tolist(),
                "teacher_rgb_mae": teacher_rgb.tolist(),
                "candidate_rgb_mae": candidate_rgb.tolist(),
            }

    group_summaries = {}
    for group_name in candidate_manifest["groups"]:
        selected = [row for row in rows if row["group"] == group_name]
        group_summaries[group_name] = {
            "agent": selected[0]["agent"],
            "model": selected[0]["model"],
            "round": selected[0]["round"],
            "cases": len(selected),
            "mean_spatial_lpips": statistics.fmean(row["spatial_lpips"] for row in selected),
            "mean_tlp": statistics.fmean(row["tlp_mean"] for row in selected),
            "mean_tlp_p95": statistics.fmean(row["tlp_p95"] for row in selected),
            "worst_case_tlp_p95": max(row["tlp_p95"] for row in selected),
            "mean_excess_tlp": statistics.fmean(row["excess_tlp_mean"] for row in selected),
            "mean_excess_tlp_p95": statistics.fmean(row["excess_tlp_p95"] for row in selected),
            "mean_vbench_candidate_score": statistics.fmean(
                row["vbench_candidate_score"] for row in selected
            ),
            "mean_vbench_teacher_score": statistics.fmean(
                row["vbench_teacher_score"] for row in selected
            ),
        }

    report = {
        "version": 1,
        "input": "decoded packaged MP4 RGB frames at native resolution and frame rate",
        "metric": {
            "lpips": {"package": "0.1.4", "net": "alex", "version": "0.1"},
            "tlp": "abs(LPIPS(candidate_t,candidate_t+1)-LPIPS(teacher_t,teacher_t+1))",
            "excess_tlp": "max(0,LPIPS(candidate_t,candidate_t+1)-LPIPS(teacher_t,teacher_t+1))",
            "vbench_style": "1 - mean adjacent-frame RGB MAE normalized to [0,1]",
            "direction": "tLP/excess-tLP lower is better; VBench-style score higher is better",
        },
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "groups": group_summaries,
        "cases": rows,
    }

    with tempfile.TemporaryDirectory(prefix=".temporal-flicker-", dir=ROOT) as temporary:
        staging = Path(temporary) / OUTPUT.name
        staging.mkdir()
        (staging / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (staging / "per-transition.json").write_text(
            json.dumps(per_transition, indent=2, allow_nan=False) + "\n"
        )
        write_csv(staging / "per-video.csv", rows)
        (staging / "README.md").write_text(
            "# Temporal flicker evaluation\n\n"
            "This evaluates the packaged Codex and Claude best-round MP4s against the exact "
            "fixed teacher MP4 for every case. `summary.json` and `per-video.csv` contain "
            "per-video and group aggregates. `per-transition.json` contains every adjacent-frame "
            "measurement for locating flicker bursts.\n\n"
            "Lower tLP and Excess-tLP are better. The VBench-style RGB score is included only "
            "as a motion-confounded comparison for these dynamic videos.\n"
        )
        shutil.move(staging, OUTPUT)

    print(json.dumps(report["groups"], indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
