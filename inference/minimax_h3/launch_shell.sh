#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

: "${H3_IMAGE:?Set H3_IMAGE to the MiniMax-H3 .sqsh image}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to the Slurm account}"
if [[ -n ${H3_DIFFUSERS:-} && -n ${H3_FL2VA:-} ]]; then
    h3_diffusers=${H3_DIFFUSERS}
    h3_fl2va=${H3_FL2VA}
else
    : "${MODEL_ROOT:?Set MODEL_ROOT, or set both H3_DIFFUSERS and H3_FL2VA}"
    h3_diffusers=${MODEL_ROOT}/MiniMax-H3-diffusers
    h3_fl2va=${MODEL_ROOT}/MiniMax-H3/FL2VA
fi
outputs=${H3_OUTPUTS:-${repo_root}/outputs/minimax-h3}

mkdir -p "${outputs}"
command=("$@")
terminal=()
placement=()
if [[ -n ${INFERENCE_NODE:-} ]]; then placement=(--nodelist="${INFERENCE_NODE}"); fi
if (( $# == 0 )); then command=(/bin/bash); terminal=(--pty); fi

exec srun -A "${SLURM_ACCOUNT}" \
    "${placement[@]}" \
    --nodes=1 --ntasks=1 \
    --partition="${SLURM_PARTITION:-interactive}" \
    --gpus=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --mem="${SLURM_MEM:-240G}" \
    --time="${SLURM_TIME:-01:00:00}" \
    --no-container-mount-home \
    --container-image="${H3_IMAGE}" \
    --container-workdir=/workspace/MiniMax-H3 \
    --container-mounts="${h3_diffusers}:/weights/h3-diffusers:ro,${h3_fl2va}:/weights/h3-fl2va:ro,${outputs}:/workspace/MiniMax-H3/outputs" \
    "${terminal[@]}" "${command[@]}"
