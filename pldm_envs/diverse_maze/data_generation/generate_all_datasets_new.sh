# This file generates new datasets for the large diverse maze environments
set -euo pipefail

# Resolve the checkout root from this script.  Set HWM_PLDM_ROOT only when the
# datasets should be written to a different checkout.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${HWM_PLDM_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RENDER_WORKERS="${RENDER_WORKERS:-1}"
RENDER_GPU_IDS="${RENDER_GPU_IDS:-}"
RENDER_VERIFY_EXISTING="${RENDER_VERIFY_EXISTING:-0}"
POSTPROCESS_WORKERS="${POSTPROCESS_WORKERS:-32}"
POSTPROCESS_BATCH_EPISODES="${POSTPROCESS_BATCH_EPISODES:-32}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if ! [[ "${RENDER_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RENDER_WORKERS must be a positive integer, got: ${RENDER_WORKERS}" >&2
    exit 1
fi

if ! [[ "${RENDER_VERIFY_EXISTING}" =~ ^[01]$ ]]; then
    echo "RENDER_VERIFY_EXISTING must be 0 or 1, got: ${RENDER_VERIFY_EXISTING}" >&2
    exit 1
fi

for value_name in POSTPROCESS_WORKERS POSTPROCESS_BATCH_EPISODES; do
    if ! [[ "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got: ${!value_name}" >&2
        exit 1
    fi
done

RENDER_VERIFY_ARGS=()
if (( RENDER_VERIFY_EXISTING )); then
    RENDER_VERIFY_ARGS+=(--verify_existing)
fi

RENDER_GPU_LIST=()
if [[ -n "${RENDER_GPU_IDS}" ]]; then
    IFS=',' read -r -a RENDER_GPU_LIST <<< "${RENDER_GPU_IDS}"
    for gpu_id in "${RENDER_GPU_LIST[@]}"; do
        if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
            echo "RENDER_GPU_IDS must be comma-separated GPU indices, got: ${RENDER_GPU_IDS}" >&2
            exit 1
        fi
    done
fi

if ! "${PYTHON_BIN}" -c "import d4rl.pointmaze"; then
    echo "D4RL/MuJoCo cannot be imported. Verify MUJOCO_PY_MUJOCO_PATH and LD_LIBRARY_PATH before generating data." >&2
    exit 1
fi

# Generate dataset for 25maps setting.
"${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/generate_data.py" --output_path "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps" --config "${PROJECT_ROOT}/pldm_envs/diverse_maze/configs/maze2d_large/25maps.yaml"

# Generate dataset for OOD evaluation (probe) setting.
"${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/generate_data.py" --output_path "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe" --config "${PROJECT_ROOT}/pldm_envs/diverse_maze/configs/maze2d_large/probe.yaml" --exclude_map_path "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps/train_maps.pt"

DATA_PATHS=(
    "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps"
    "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe"
)

render_dataset() {
    local data_path="$1"
    local worker_id
    local pid
    local gpu_id
    local pids=()

    if (( RENDER_WORKERS == 1 )); then
        "${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/render_data.py" \
            --data_path "${data_path}" \
            ${RENDER_VERIFY_ARGS[@]+"${RENDER_VERIFY_ARGS[@]}"}
    else
        echo "Rendering ${data_path} with ${RENDER_WORKERS} worker processes"
        # Background workers otherwise outlive this script when it receives
        # Ctrl+C.  Install the handler before starting them so an interrupted
        # rendering run can be resumed without orphaned renderers.
        trap 'echo "Interrupt received; stopping rendering workers..." >&2; kill "${pids[@]}" 2>/dev/null || true; wait "${pids[@]}" 2>/dev/null || true; exit 130' INT TERM
        for worker_id in $(seq 0 $((RENDER_WORKERS - 1))); do
            if (( ${#RENDER_GPU_LIST[@]} )); then
                gpu_id="${RENDER_GPU_LIST[$((worker_id % ${#RENDER_GPU_LIST[@]}))]}"
                echo "  worker ${worker_id} -> GPU ${gpu_id}"
                CUDA_VISIBLE_DEVICES="${gpu_id}" MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
                    "${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/render_data.py" \
                    --data_path "${data_path}" \
                    --workers_num "${RENDER_WORKERS}" \
                    --worker_id "${worker_id}" \
                    ${RENDER_VERIFY_ARGS[@]+"${RENDER_VERIFY_ARGS[@]}"} &
            else
                "${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/render_data.py" \
                    --data_path "${data_path}" \
                    --workers_num "${RENDER_WORKERS}" \
                    --worker_id "${worker_id}" \
                    ${RENDER_VERIFY_ARGS[@]+"${RENDER_VERIFY_ARGS[@]}"} &
            fi
            pids+=("$!")
        done

        for pid in "${pids[@]}"; do
            if ! wait "${pid}"; then
                echo "A rendering worker failed; stopping the remaining workers." >&2
                for worker_id in "${pids[@]}"; do
                    kill "${worker_id}" 2>/dev/null || true
                done
                wait || true
                return 1
            fi
        done
        trap - INT TERM
    fi

    "${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/postprocess_images.py" \
        --data_path "${data_path}" \
        --num_workers "${POSTPROCESS_WORKERS}" \
        --batch_episodes "${POSTPROCESS_BATCH_EPISODES}"
}

# Render each dataset, then combine its PNGs into images.npy.
for DATA_PATH in "${DATA_PATHS[@]}"; do
    render_dataset "${DATA_PATH}"
done

# Generate OOD evaluation trials for the 5 maps setting
"${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/evaluation/generate_starts_targets.py" --data_path "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe"
