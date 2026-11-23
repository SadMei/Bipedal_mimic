# Usage: source /path/to/Bipedal_mimic/env.sh
BIPEDAL_MIMIC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
source "$BIPEDAL_MIMIC_ROOT/.venv/bin/activate" || return 1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$BIPEDAL_MIMIC_ROOT/../.cache/uv}"
export UV_HTTP_TIMEOUT=600
export UV_LINK_MODE=hardlink
