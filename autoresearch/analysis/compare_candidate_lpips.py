"""Compare two saved MiniMax-H3 RGB output sets frame by frame with LPIPS."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys


here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "harbor"))
from verify import METRIC, metric_frame


parser = argparse.ArgumentParser()
parser.add_argument("--left", type=Path, required=True)
parser.add_argument("--right", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--id-start", type=int, default=1,
                    help="Numeric suffix for the first of the four MiniMax-H3 cases")
parser.add_argument("--count", type=int, default=4)
args = parser.parse_args()

import torch
import lpips

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this LPIPS comparison")
if importlib.metadata.version("lpips") != METRIC["package"]:
    raise RuntimeError("LPIPS package version does not match the benchmark metric")

metric = lpips.LPIPS(net="alex", version="0.1").eval().cuda()
rows = []
with torch.inference_mode():
    for index in range(args.count):
        left = torch.load(args.left / f"output-0-{index}.pt", map_location="cpu",
                          weights_only=True, mmap=True)
        right = torch.load(args.right / f"output-0-{index}.pt", map_location="cpu",
                           weights_only=True, mmap=True)
        if left["video"].shape != right["video"].shape:
            raise ValueError(f"Video shape mismatch for case {index + 1}")
        frames = left["video"].shape[2]
        distances = [
            metric(metric_frame(left["video"], frame, "minimax_h3").cuda(),
                   metric_frame(right["video"], frame, "minimax_h3").cuda()).item()
            for frame in range(frames)
        ]
        rows.append({"id": f"minimax_h3-{args.id_start + index:02d}", "frames": frames,
                     "lpips": statistics.mean(distances)})

result = {
    "metric": "LPIPS alex v0.1, decoded RGB, frame mean",
    "left": str(args.left),
    "right": str(args.right),
    "cases": rows,
    "lpips_mean": statistics.mean(row["lpips"] for row in rows),
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
