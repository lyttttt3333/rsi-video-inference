#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
round=${1:?round}
(( round >= 1 && round <= 50 ))
root=$(cd -- "$here/../../.." && pwd)
: "${SANA_IMAGE:?Set SANA_IMAGE to the SANA-Video .sqsh image}"
: "${SANA_CHECKPOINT_ROOT:?Set SANA_CHECKPOINT_ROOT}"
: "${SANA_VAE:?Set SANA_VAE}"
: "${SANA_GEMMA_REPO:?Set SANA_GEMMA_REPO}"
: "${SANA_REFERENCE:?Set SANA_REFERENCE to the eight-case teacher directory}"
: "${METRIC_ASSETS:?Set METRIC_ASSETS to prepared LPIPS assets}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to the Slurm account}"
output=$here/runs/round-$(printf '%02d' "$round")
workspace=$here/workspace
cases=$here/cases.json
reference=$SANA_REFERENCE
image=$SANA_IMAGE
gemma_snapshot=${SANA_GEMMA_SNAPSHOT:-$SANA_GEMMA_REPO/snapshots/569d9809d0c8b6722d4d31b5a77a2ec7a400650a}
mounts="$SANA_CHECKPOINT_ROOT:/weights/sana-checkpoint:ro,$SANA_VAE:/weights/ltx-vae:ro,$gemma_snapshot:/weights/gemma/snapshots/current:ro,$SANA_GEMMA_REPO/blobs:/weights/gemma/blobs:ro"
[[ -f $workspace/candidate.py && -f $cases && -f $reference/manifest.json ]]
mkdir -p "$output"
srun -A "$SLURM_ACCOUNT" --partition="${SLURM_PARTITION:-interactive}" --constraint="${GPU_CONSTRAINT:-H100}" \
  --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=128G \
  --time=00:14:00 --job-name="claude-sana-r$round" \
  --no-container-mount-home --container-image="$image" --container-workdir=/submission \
  --container-mounts="$mounts,$root/benchmark/common:/harbor:ro,$workspace:/submission:ro,$cases:/cases.json:ro,$reference:/reference:ro,$METRIC_ASSETS:/metric-assets:ro,$output:/round-output" \
  env PYTHONPATH=/metric-assets/python TORCH_HOME=/metric-assets/torch OMP_NUM_THREADS=8 \
  python /harbor/verify.py --model sana_video2 --submission /submission \
    --weights /workspace/SANA-Video-2.0/weights --cases /cases.json \
    --reference /reference --output /round-output/eval --repeats 1 --timeout 780
cp "$output/eval/reward.json" "$output/reward.json"
