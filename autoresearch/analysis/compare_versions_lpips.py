"""Compare multiple saved H3 output sets against one decoded-RGB reference set."""
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
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--candidate", action="append", required=True,
                    help="LABEL=PATH; may be supplied more than once")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

import torch
import lpips

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this LPIPS comparison")
if importlib.metadata.version("lpips") != METRIC["package"]:
    raise RuntimeError("LPIPS package version does not match the benchmark metric")

metric = lpips.LPIPS(net="alex", version="0.1").eval().cuda()
versions = []
with torch.inference_mode():
    for specification in args.candidate:
        label, separator, raw_path = specification.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"Invalid candidate specification: {specification!r}")
        candidate_path = Path(raw_path)
        rows = []
        for index in range(4):
            reference = torch.load(args.reference / f"output-0-{index}.pt", map_location="cpu",
                                   weights_only=True, mmap=True)
            candidate = torch.load(candidate_path / f"output-0-{index}.pt", map_location="cpu",
                                   weights_only=True, mmap=True)
            if reference["video"].shape != candidate["video"].shape:
                raise ValueError(f"Video shape mismatch for {label}, case {index + 1}")
            frames = reference["video"].shape[2]
            distances = [
                metric(metric_frame(reference["video"], frame, "minimax_h3").cuda(),
                       metric_frame(candidate["video"], frame, "minimax_h3").cuda()).item()
                for frame in range(frames)
            ]
            rows.append({"id": f"minimax_h3-{index + 1:02d}", "frames": frames,
                         "lpips": statistics.mean(distances)})
        versions.append({"label": label, "path": str(candidate_path), "cases": rows,
                         "lpips_mean": statistics.mean(row["lpips"] for row in rows)})

result = {
    "metric": "LPIPS alex v0.1, decoded RGB, frame mean",
    "reference": str(args.reference),
    "versions": versions,
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
