#!/usr/bin/env bash
##############################################################################
# BindCraft – unified installer for Kaggle (old layout, new dependency handling)
# Author: <your-name>                  Date: 2025-05-27
# Tested on: Kaggle Python 3.10 CPU & GPU (CUDA 11.8) images – May 2025
##############################################################################
set -euo pipefail
IFS=$'\n\t'

######################### command-line options ################################
CUDA_VERSION=''                     # --cuda <11.8> for GPU build
CPU_ONLY=false                      # --cpu           for CPU-only

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) CUDA_VERSION="$2"; shift 2 ;;
    --cpu)  CPU_ONLY=true;      shift   ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$CPU_ONLY" == true && -n "$CUDA_VERSION" ]]; then
  echo "Error: specify either --cpu or --cuda, not both" >&2
  exit 1
fi

###################### micromamba bootstrap & setup ###########################
MICROMAMBA_DIR=/tmp/micromamba
ENV_DIR=/tmp/bindcraft_env

if [[ ! -x $MICROMAMBA_DIR/bin/micromamba ]]; then
  echo "Installing micromamba to $MICROMAMBA_DIR..."
  mkdir -p "$MICROMAMBA_DIR"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "$MICROMAMBA_DIR" bin/micromamba
  chmod +x "$MICROMAMBA_DIR/bin/micromamba"
fi

MICROMAMBA_BIN="$MICROMAMBA_DIR/bin/micromamba"
export PATH="$MICROMAMBA_DIR/bin:$PATH"

########################### package specification #############################
COMMON_PKGS=(
  python=3.10
  pip pandas matplotlib "numpy<2.0.0"
  biopython scipy pdbfixer seaborn libgfortran5 tqdm
  jupyter ffmpeg pyrosetta fsspec py3dmol chex
  dm-haiku "flax<0.10.0" dm-tree joblib
  ml-collections immutabledict optax
)

if [[ "$CPU_ONLY" == true || -z "$CUDA_VERSION" ]]; then
  echo "Preparing CPU-only environment"
  ALL_PKGS=( "${COMMON_PKGS[@]}" jaxlib jax )
  CHANNELS=( -c conda-forge --channel https://conda.graylab.jhu.edu )
else
  echo "Preparing GPU environment (CUDA ${CUDA_VERSION})"
  GPU_PKGS=( "jaxlib=*=*cuda*" jax cuda-nvcc cudnn )
  ALL_PKGS=( "${COMMON_PKGS[@]}" "${GPU_PKGS[@]}" )
  CHANNELS=( -c conda-forge -c nvidia --channel https://conda.graylab.jhu.edu )
fi

############################# environment creation ############################
echo "Solving environment; this may take several minutes..."
"$MICROMAMBA_BIN" create -y -p "$ENV_DIR" "${CHANNELS[@]}" "${ALL_PKGS[@]}"

############################ integrity check ##################################
echo "Verifying installation of core packages..."
"$MICROMAMBA_BIN" run -p "$ENV_DIR" python - <<'PYCODE'
import importlib, sys
core = ["numpy","scipy","jax","jaxlib","pandas","matplotlib"]
missing = [pkg for pkg in core if importlib.util.find_spec(pkg) is None]
if missing:
    print("Missing packages:", missing, file=sys.stderr)
    sys.exit(1)
sys.exit(0)
PYCODE

############################ ColabDesign install ##############################
echo "Installing ColabDesign (pip, no deps)..."
"$MICROMAMBA_BIN" run -p "$ENV_DIR" pip install --no-deps \
    git+https://github.com/sokrypton/ColabDesign.git
"$MICROMAMBA_BIN" run -p "$ENV_DIR" python -c "import colabdesign"

########################## AlphaFold weights setup ############################
echo "Downloading and extracting AlphaFold2 parameters..."
WEIGHTS_DIR=/tmp/alphafold
SYMLINK_DIR="$ENV_DIR/params"
ARCHIVE=/tmp/alphafold_params_2022-12-06.tar

mkdir -p "$WEIGHTS_DIR" "$SYMLINK_DIR"
wget -q --show-progress -O "$ARCHIVE" \
     https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf "$ARCHIVE" -C "$WEIGHTS_DIR"
rm "$ARCHIVE"

# Symlink each params file into the environment
for file in "$WEIGHTS_DIR"/*; do
  ln -sf "$file" "$SYMLINK_DIR"/
done

######################### executable permissions ##############################
chmod +x "$(pwd)/functions/dssp"           2>/dev/null || true
chmod +x "$(pwd)/functions/DAlphaBall.gcc"  2>/dev/null || true

############################# clean-up ########################################
echo "Cleaning package cache..."
"$MICROMAMBA_BIN" clean -a -y

############################## summary ########################################
elapsed=$SECONDS
printf "\nInstallation complete.\n"
printf "Environment path: %s\n" "$ENV_DIR"
if [[ -n "$CUDA_VERSION" ]]; then
  printf "GPU support: CUDA %s\n" "$CUDA_VERSION"
else
  printf "GPU support: none (CPU-only)\n"
fi
printf "Elapsed time: %d h %d m %d s\n" \
       "$((elapsed/3600))" "$(((elapsed/60)%60))" "$((elapsed%60))"
