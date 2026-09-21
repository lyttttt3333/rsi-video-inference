#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
round=${1:?round}
(( round >= 1 && round <= 25 ))
root=$(cd -- "$here/../../.." && pwd)
: "${MODEL_ROOT:?Set MODEL_ROOT to the local model directory}"
: "${H3_IMAGE:?Set H3_IMAGE to the MiniMax-H3 .sqsh image}"
: "${H3_PUBLIC_REFERENCE:?Set H3_PUBLIC_REFERENCE to the public teacher directory}"
: "${METRIC_ASSETS:?Set METRIC_ASSETS to prepared LPIPS assets}"
: "${SOL_SOURCE:?Set SOL_SOURCE to the sparse backend source directory}"
: "${TOOLCHAIN_MOUNTS:?Set TOOLCHAIN_MOUNTS to required read-only compiler mounts}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to the Slurm account}"
models=$MODEL_ROOT
output=$here/runs/round-$(printf '%02d' "$round")
workspace=$here/workspace
cases=$here/cases.json
reference=$H3_PUBLIC_REFERENCE
image=$H3_IMAGE
app=/workspace/MiniMax-H3
fused_transformer=${H3_FUSED_TRANSFORMER:-$models/MiniMax-H3-transformer-fused-qkv}
sol_source=$SOL_SOURCE
cache_root=$here/cache

[[ -f $workspace/candidate.py && -f $cases && -f $reference/manifest.json ]]
[[ ! -e $output ]] || { echo "Refusing to overwrite $output" >&2; exit 2; }
mkdir -p "$output" "$cache_root"/{adaln,triton,torchinductor,cuda}

mounts="$models/MiniMax-H3-diffusers:/weights/h3-diffusers:ro,$models/MiniMax-H3/FL2VA:/weights/h3-fl2va:ro"
if [[ -d $fused_transformer ]]; then
  mounts+=",$fused_transformer:/weights/h3-fused-transformer:ro"
fi
toolchain=$TOOLCHAIN_MOUNTS

srun -A "$SLURM_ACCOUNT" --partition="${SLURM_PARTITION:-interactive}" --constraint="${GPU_CONSTRAINT:-H100}" \
  --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem=240G \
  --time=00:28:00 --job-name="claude-h3-r$round" \
  --no-container-mount-home --container-image="$image" --container-workdir=/submission \
  --container-mounts="$mounts,$toolchain,$sol_source:/sol-src:ro,$cache_root:/persistent-cache,$root/benchmark/common:/harbor:ro,$workspace:/submission:ro,$cases:/cases.json:ro,$reference:/reference:ro,$METRIC_ASSETS:/metric-assets:ro,$output:/round-output" \
  env CC=/usr/bin/gcc PYTHONPATH=/sol-src:/metric-assets/python \
    TORCH_HOME=/metric-assets/torch OMP_NUM_THREADS=8 \
    H3_FUSED_TRANSFORMER=/weights/h3-fused-transformer \
    H3_ADALN_CACHE_DIR=/persistent-cache/adaln \
    TRITON_CACHE_DIR=/persistent-cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/persistent-cache/torchinductor \
    CUDA_CACHE_PATH=/persistent-cache/cuda \
  python /harbor/verify.py --model minimax_h3 --submission /submission \
    --weights "$app/weights" --cases /cases.json --reference /reference \
    --output /round-output/eval --repeats 1 --timeout 1560
cp "$output/eval/reward.json" "$output/reward.json"
