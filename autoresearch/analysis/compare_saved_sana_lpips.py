"""Compare two saved decoded SANA tensors frame by frame with LPIPS."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harbor"))
from verify import METRIC, metric_frame


parser = argparse.ArgumentParser()
parser.add_argument("--left", type=Path, required=True)
parser.add_argument("--right", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--id", required=True)
args = parser.parse_args()

import torch
import lpips

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for LPIPS")
if importlib.metadata.version("lpips") != METRIC["package"]:
    raise RuntimeError("LPIPS package version does not match the benchmark metric")

left = torch.load(args.left, map_location="cpu", weights_only=True, mmap=True)
right = torch.load(args.right, map_location="cpu", weights_only=True, mmap=True)
if left["video"].shape != right["video"].shape:
    raise ValueError(f"Video shape mismatch: {left['video'].shape} != {right['video'].shape}")

metric = lpips.LPIPS(net="alex", version="0.1").eval().cuda()
frames = left["video"].shape[2]
distances = []
with torch.inference_mode():
    for frame in range(frames):
        a = metric_frame(left["video"], frame, "sana_video2").cuda()
        b = metric_frame(right["video"], frame, "sana_video2").cuda()
        distances.append(metric(a, b).item())

result = {
    "id": args.id,
    "metric": "LPIPS alex v0.1, decoded RGB, mean over all frames",
    "left": str(args.left),
    "right": str(args.right),
    "shape": list(left["video"].shape),
    "frames": frames,
    "lpips": statistics.mean(distances),
    "frame_lpips_min": min(distances),
    "frame_lpips_max": max(distances),
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
