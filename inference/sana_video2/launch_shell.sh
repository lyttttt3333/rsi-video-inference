#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

: "${SANA_IMAGE:?Set SANA_IMAGE to the SANA-Video .sqsh image}"
: "${SANA_CHECKPOINT_ROOT:?Set SANA_CHECKPOINT_ROOT to the checkpoint directory}"
: "${SANA_VAE:?Set SANA_VAE to the LTX VAE repository}"
: "${SANA_GEMMA_REPO:?Set SANA_GEMMA_REPO to the local Gemma cache repository}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT to the Slurm account}"
checkpoint=${SANA_CHECKPOINT_ROOT}
vae=${SANA_VAE}
gemma_repo=${SANA_GEMMA_REPO}
gemma_snapshot=${SANA_GEMMA_SNAPSHOT:-${gemma_repo}/snapshots/569d9809d0c8b6722d4d31b5a77a2ec7a400650a}
outputs=${SANA_OUTPUTS:-${repo_root}/outputs/sana-video2}

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
    --mem="${SLURM_MEM:-128G}" \
    --time="${SLURM_TIME:-00:30:00}" \
    --no-container-mount-home \
    --container-image="${SANA_IMAGE}" \
    --container-workdir=/workspace/SANA-Video-2.0 \
    --container-mounts="${checkpoint}:/weights/sana-checkpoint:ro,${vae}:/weights/ltx-vae:ro,${gemma_snapshot}:/weights/gemma/snapshots/current:ro,${gemma_repo}/blobs:/weights/gemma/blobs:ro,${outputs}:/workspace/SANA-Video-2.0/outputs" \
    "${terminal[@]}" "${command[@]}"
