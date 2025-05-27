#!/usr/bin/env bash
##############################################################################
# BindCraft – Kaggle-compatible installer (prefix-based layout)
# Author: <your-name>          Date: 2025-05-27
# Usage: ./install_bindcraft.sh [--cuda <version>] [--cpu]
##############################################################################
set -euo pipefail
IFS=$'\n\t'

########################## Parse command‐line flags ###########################
CUDA_VERSION=""        # e.g. "--cuda 11.8"
CPU_ONLY=false         # "--cpu" for CPU‐only build

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      CUDA_VERSION="$2"
      shift 2
      ;;
    --cpu)
      CPU_ONLY=true
      shift
      ;;
    *)
      echo "Usage: $0 [--cuda <version>] [--cpu]" >&2
      exit 1
      ;;
  esac
done

if [[ "$CPU_ONLY" == true && -n "$CUDA_VERSION" ]]; then
  echo "Error: specify either --cpu or --cuda, not both" >&2
  exit 1
fi

########################## Directory configuration ###########################
MICROMAMBA_DIR=/tmp/micromamba
ENV_DIR=/tmp/bindcraft_env

####################### Bootstrap standalone micromamba #######################
if ! command -v micromamba &> /dev/null; then
  echo "📥 Bootstrapping micromamba into $MICROMAMBA_DIR"
  mkdir -p "$MICROMAMBA_DIR"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "$MICROMAMBA_DIR" bin/micromamba
  chmod +x "$MICROMAMBA_DIR/bin/micromamba"
fi

MICROMAMBA_BIN="$MICROMAMBA_DIR/bin/micromamba"
export PATH="$MICROMAMBA_DIR/bin:$PATH"

############################ Package specification ############################
BASE_CHANNELS=(-c conda-forge --channel https://conda.graylab.jhu.edu)
GPU_CHANNELS=(-c nvidia)

COMMON_PKGS=(
  python=3.10
  pip               # ensure pip is present
  pandas matplotlib "numpy<2.0.0"
  biopython scipy pdbfixer
  seaborn libgfortran5 tqdm
  jupyter ffmpeg pyrosetta
  fsspec py3dmol chex
  dm-haiku "flax<0.10.0" dm-tree
  joblib ml-collections immutabledict optax
)

if [[ "$CPU_ONLY" == true || -z "$CUDA_VERSION" ]]; then
  echo "⚙️  Preparing CPU‐only environment"
  ALL_PKGS=("${COMMON_PKGS[@]}" jax jaxlib)
  CHANNELS=("${BASE_CHANNELS[@]}")
else
  echo "⚙️  Preparing GPU environment (CUDA $CUDA_VERSION)"
  # We’ll install JAX with CUDA support via pip later; for now include CUDA toolkits
  ALL_PKGS=("${COMMON_PKGS[@]}" cuda-nvcc cudnn)
  CHANNELS=("${BASE_CHANNELS[@]}" "${GPU_CHANNELS[@]}")
fi

############################ Create conda environment #########################
echo "🚧 Creating Micromamba environment at $ENV_DIR"
"$MICROMAMBA_BIN" create -y -p "$ENV_DIR" "${CHANNELS[@]}" "${ALL_PKGS[@]}"

############################ Install JAX via pip ##############################
echo "🚀 Installing JAX into the environment"
if [[ -n "$CUDA_VERSION" ]]; then
  # Determine major version for JAX CUDA wheel tag
  CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
  "$MICROMAMBA_BIN" run -p "$ENV_DIR" pip install --upgrade \
    "jax[cuda${CUDA_MAJOR}]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
  "$MICROMAMBA_BIN" run -p "$ENV_DIR" pip install --upgrade jax jaxlib
fi

# verify JAX import
"$MICROMAMBA_BIN" run -p "$ENV_DIR" python - <<'PYCODE'
import importlib.util, sys
for pkg in ("jax","jaxlib"):
    if importlib.util.find_spec(pkg) is None:
        print(f"Missing {pkg}", file=sys.stderr)
        sys.exit(1)
sys.exit(0)
PYCODE

########################## Install ColabDesign ################################
echo "📦 Installing ColabDesign (no dependencies)"
"$MICROMAMBA_BIN" run -p "$ENV_DIR" pip install --no-deps \
    git+https://github.com/sokrypton/ColabDesign.git
"$MICROMAMBA_BIN" run -p "$ENV_DIR" python -c "import colabdesign" \
    || { echo "❌ ColabDesign import failed"; exit 1; }

#################### Download & symlink AlphaFold2 weights ####################
echo "📥 Downloading AlphaFold2 parameters"
WEIGHTS_DIR=/tmp/alphafold
SYMLINK_DIR="$ENV_DIR/params"
ARCHIVE=/tmp/alphafold_params_2022-12-06.tar

mkdir -p "$WEIGHTS_DIR" "$SYMLINK_DIR"
wget -q --show-progress -O "$ARCHIVE" \
     https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf "$ARCHIVE" -C "$WEIGHTS_DIR"
rm "$ARCHIVE"

echo "🔗 Creating symlinks in $SYMLINK_DIR"
for f in "$WEIGHTS_DIR"/*; do
  ln -sf "$f" "$SYMLINK_DIR"/
done

######################### Set executable permissions ##########################
chmod +x "$(pwd)/functions/dssp"           2>/dev/null || true
chmod +x "$(pwd)/functions/DAlphaBall.gcc"  2>/dev/null || true

############################## Clean up cache #################################
echo "🧹 Cleaning Micromamba cache"
"$MICROMAMBA_BIN" clean -a -y

############################### Done & summary ################################
ELAPSED=$SECONDS
printf "\n✅  Installation complete\n"
printf "▶ Environment path: %s\n" "$ENV_DIR"
if [[ -n "$CUDA_VERSION" ]]; then
  printf "▶ GPU support: CUDA %s\n" "$CUDA_VERSION"
else
  printf "▶ GPU support: none (CPU-only)\n"
fi
printf "▶ To run commands inside the env:\n"
printf "   %s run -p %s <command>\n" "$MICROMAMBA_BIN" "$ENV_DIR"
printf "⌛ Elapsed time: %d h %d m %d s\n" \
       "$((ELAPSED/3600))" "$(((ELAPSED/60)%60))" "$((ELAPSED%60))"
