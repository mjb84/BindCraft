#!/usr/bin/env bash
##############################################################################
# BindCraft – unified installer (old layout, new dependency handling)
# Author: <your-name>                  Date: 2025-05-27
# Tested on: Kaggle Python 3.10 CPU & GPU (CUDA 11.8) images – May 2025
##############################################################################
set -euo pipefail
IFS=$'\n\t'

############################ command-line options ############################
CUDA_VERSION=''                     # --cuda <11.8>      for GPU build
CPU_ONLY=false                      # --cpu              for CPU build

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) CUDA_VERSION="$2"; shift 2 ;;
    --cpu)  CPU_ONLY=true;      shift   ;;
    *) echo "Unknown option $1" >&2; exit 1 ;;
  esac
done
[[ -n "$CUDA_VERSION" && "$CPU_ONLY" == true ]] && {
  echo "Specify either --cuda or --cpu, not both" >&2; exit 1; }

############################ micromamba bootstrap ############################
MICROMAMBA_DIR=/tmp/micromamba
ENV_DIR=/tmp/bindcraft_env

if [[ ! -x $MICROMAMBA_DIR/micromamba ]]; then
  echo "→ Installing micromamba ..."
  mkdir -p "$MICROMAMBA_DIR"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest |
    tar -xvj -C "$MICROMAMBA_DIR" bin/micromamba
  chmod +x "$MICROMAMBA_DIR/bin/micromamba"
fi
export PATH="$MICROMAMBA_DIR/bin:$PATH"

############################ package specification ###########################
COMMON_PKGS=(
  python=3.10 pip pandas matplotlib "numpy<2.0.0" biopython scipy pdbfixer
  seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta
  fsspec py3dmol chex dm-haiku "flax<0.10.0" dm-tree joblib
  ml-collections immutabledict optax
)

if [[ "$CPU_ONLY" == true ]] || [[ -z "$CUDA_VERSION" ]]; then
  # ---------------- CPU build ----------------
  echo "→ Preparing CPU-only environment"
  ALL_PKGS=( "${COMMON_PKGS[@]}" jaxlib jax )
  CHANNELS=( -c conda-forge --channel https://conda.graylab.jhu.edu )
else
  # ---------------- GPU build ----------------
  echo "→ Preparing GPU environment (CUDA ${CUDA_VERSION})"
  GPU_PKGS=( "jaxlib=*=*cuda*" jax cuda-nvcc cudnn )
  ALL_PKGS=( "${COMMON_PKGS[@]}" "${GPU_PKGS[@]}" )
  CHANNELS=( -c conda-forge -c nvidia --channel https://conda.graylab.jhu.edu )
fi

############################ create environment ##############################
echo "→ Solving environment … this can take a few minutes"
micromamba create -y -p "$ENV_DIR" "${CHANNELS[@]}" "${ALL_PKGS[@]}"

############################ integrity check #################################
echo "→ Verifying core packages"
micromamba run -p "$ENV_DIR" python - <<'PY'
import importlib, sys
core = ["numpy","scipy","jax","jaxlib","pandas","matplotlib"]
missing=[m for m in core if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PY

############################ ColabDesign #####################################
echo "→ Installing ColabDesign (pip, no deps)"
micromamba run -p "$ENV_DIR" pip install --no-deps \
    git+https://github.com/sokrypton/ColabDesign.git
micromamba run -p "$ENV_DIR" python -c "import colabdesign"

############################ AlphaFold weights ###############################
echo "→ Fetching AlphaFold weights"
WEIGHTS_DIR=/tmp/alphafold
SYMLINK_DIR="$ENV_DIR/params"
ARCHIVE=/tmp/alphafold_params_2022-12-06.tar

mkdir -p "$WEIGHTS_DIR" "$SYMLINK_DIR"
wget -q --show-progress -O "$ARCHIVE" \
     https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf "$ARCHIVE" -C "$WEIGHTS_DIR"
rm "$ARCHIVE"
# symlink into env for ColabDesign-style discovery
ln -sf "$WEIGHTS_DIR"/* "$SYMLINK_DIR"/
[[ -f "$SYMLINK_DIR/params_model_5_ptm.npz" ]]

############################ executable perms ################################
chmod +x "$(pwd)/functions/dssp"           2>/dev/null || true
chmod +x "$(pwd)/functions/DAlphaBall.gcc" 2>/dev/null || true

############################ clean-up ########################################
micromamba clean -a -y

############################ summary #########################################
elapsed=$SECONDS
printf "\n✅  BindCraft installation complete\n"
printf "📦  Environment path:  %s\n" "$ENV_DIR"
printf "⚡  GPU support:       %s\n" "${CUDA_VERSION:-CPU-only}"
printf "⏱  Elapsed time:      %d h %d m %d s\n" \
       "$((elapsed/3600))" "$(((elapsed/60)%60))" "$((elapsed%60))"
