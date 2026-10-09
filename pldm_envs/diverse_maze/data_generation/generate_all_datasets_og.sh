# This file downloads the original datasets and render them
set -euo pipefail

# Resolve the checkout root from this script.  Set HWM_PLDM_ROOT only when the
# datasets should be written to a different checkout.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${HWM_PLDM_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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

# render the datasets. save images as numpy
for DATA_PATH in "${DATA_PATHS[@]}"; do
    "${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/render_data.py" --data_path "$DATA_PATH"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/pldm_envs/diverse_maze/data_generation/postprocess_images.py" --data_path "$DATA_PATH"
done
