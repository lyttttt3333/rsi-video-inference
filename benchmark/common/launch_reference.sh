#!/usr/bin/env bash
# Owner-side public calibration/smoke launcher. It never accesses private cases.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$here/../.." && pwd)
model=${1:?minimax_h3 or sana_video2}
mode=${2:-smoke}
[[ $mode == smoke || $mode == calibrate ]] || exit 2
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to the Slurm account}"
: "${METRIC_ASSETS:?Set METRIC_ASSETS to the prepared offline metric directory}"

if [[ $model == minimax_h3 ]]; then
  : "${H3_IMAGE:?Set H3_IMAGE to the MiniMax-H3 .sqsh image}"
  : "${MODEL_ROOT:?Set MODEL_ROOT to the local model directory}"
  slug=minimax-h3-hot-inference
  output_key=minimax_h3_native_dense
  image=$H3_IMAGE
  memory=240G
  weights=/workspace/MiniMax-H3/weights
  mounts="$MODEL_ROOT/MiniMax-H3-diffusers:/weights/h3-diffusers:ro,$MODEL_ROOT/MiniMax-H3/FL2VA:/weights/h3-fl2va:ro"
elif [[ $model == sana_video2 ]]; then
  : "${SANA_IMAGE:?Set SANA_IMAGE to the SANA-Video .sqsh image}"
  : "${SANA_CHECKPOINT_ROOT:?Set SANA_CHECKPOINT_ROOT}"
  : "${SANA_VAE:?Set SANA_VAE}"
  : "${SANA_GEMMA_REPO:?Set SANA_GEMMA_REPO}"
  slug=sana-video2-hot-inference
  output_key=sana_video2_gpu_resident_all8
  image=$SANA_IMAGE
  memory=128G
  weights=/workspace/SANA-Video-2.0/weights
  gemma_snapshot=${SANA_GEMMA_SNAPSHOT:-$SANA_GEMMA_REPO/snapshots/569d9809d0c8b6722d4d31b5a77a2ec7a400650a}
  mounts="$SANA_CHECKPOINT_ROOT:/weights/sana-checkpoint:ro,$SANA_VAE:/weights/ltx-vae:ro,$gemma_snapshot:/weights/gemma/snapshots/current:ro,$SANA_GEMMA_REPO/blobs:/weights/gemma/blobs:ro"
else
  exit 2
fi

task=$repo_root/benchmark/tasks/$slug
cases=$task/environment/validation/cases.json
split=public
if [[ $model == sana_video2 ]]; then
  cases=$task/environment/validation/all-cases.json
  split=all
fi
output=${REFERENCE_OUTPUT_ROOT:-$repo_root/outputs/harbor}/$output_key
mkdir -p "$output"
run=$(mktemp -d "$output/$mode-$split-XXXXXXXX")
echo "Owner preparation output: $run"

srun -A "$SLURM_ACCOUNT" -p "${SLURM_PARTITION:-interactive}" --constraint="${GPU_CONSTRAINT:-H100}" \
  --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem="$memory" \
  --time="${REFERENCE_TIME:-01:30:00}" --job-name="teacher-$model-$mode" \
  --no-container-mount-home --container-image="$image" --container-workdir=/runner \
  --container-mounts="$mounts,$here:/runner:ro,$METRIC_ASSETS:/metric-assets:ro,$task/environment/baseline/source:/submission:ro,$cases:/cases.json:ro,$run:/run-output,$output:/reference-output" \
  env PYTHONPATH=/metric-assets/python TORCH_HOME=/metric-assets/torch OMP_NUM_THREADS=8 \
  python /runner/reference_driver.py --model "$model" --mode "$mode" --split "$split" --weights "$weights"

