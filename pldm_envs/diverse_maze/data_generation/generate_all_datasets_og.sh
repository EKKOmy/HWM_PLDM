# This file downloads the original datasets and render them
set -euo pipefail

# Resolve the checkout root from this script.  Set HWM_PLDM_ROOT only when the
# datasets should be written to a different checkout.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${HWM_PLDM_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RENDER_WORKERS="${RENDER_WORKERS:-1}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if ! [[ "${RENDER_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RENDER_WORKERS must be a positive integer, got: ${RENDER_WORKERS}" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import huggingface_hub"; then
    echo "Missing dependency: huggingface_hub. Run: ${PYTHON_BIN} -m pip install huggingface_hub==1.6.0" >&2
    exit 1
fi

# Download datasets from HF into pldm_envs/diverse_maze/datasets.
"${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/download_ds_from_hf.py" \
    --out-dir "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets"

DATA_PATHS=(
    "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_25maps"
    "${PROJECT_ROOT}/pldm_envs/diverse_maze/datasets/maze2d_large_diverse_probe"
)

for DATA_PATH in "${DATA_PATHS[@]}"; do
    if [[ ! -f "${DATA_PATH}/metadata.pt" ]]; then
        echo "Dataset download is incomplete: missing ${DATA_PATH}/metadata.pt" >&2
        exit 1
    fi
done

if ! "${PYTHON_BIN}" -c "import d4rl.pointmaze"; then
    echo "D4RL/MuJoCo cannot be imported. Verify MUJOCO_PY_MUJOCO_PATH and LD_LIBRARY_PATH before rendering." >&2
    exit 1
fi

render_dataset() {
    local data_path="$1"
    local worker_id
    local pid
    local pids=()

    if (( RENDER_WORKERS == 1 )); then
        "${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/render_data.py" \
            --data_path "${data_path}"
    else
        echo "Rendering ${data_path} with ${RENDER_WORKERS} worker processes"
        for worker_id in $(seq 0 $((RENDER_WORKERS - 1))); do
            "${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/render_data.py" \
                --data_path "${data_path}" \
                --workers_num "${RENDER_WORKERS}" \
                --worker_id "${worker_id}" &
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
    fi

    "${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/postprocess_images.py" \
        --data_path "${data_path}"
}

# Render each dataset, then combine its PNGs into images.npy.
for DATA_PATH in "${DATA_PATHS[@]}"; do
    render_dataset "${DATA_PATH}"
done
