#!/bin/bash
################## BindCraft Installation Script for Kaggle (Conda-installed JAX)

CUDA_VERSION=""
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --cuda)
      CUDA_VERSION="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--cuda <version>]"
      exit 1
      ;;
  esac
done

MICROMAMBA_DIR="/tmp/micromamba"
ENV_DIR="/tmp/bindcraft_env"

echo "Installing Micromamba…"
wget -qO micromamba.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest || exit 1
tar -xvjf micromamba.tar.bz2 bin/micromamba    || exit 1
chmod +x bin/micromamba                       || exit 1
mkdir -p $MICROMAMBA_DIR                      || exit 1
mv bin/micromamba $MICROMAMBA_DIR/            || exit 1
rm -rf micromamba.tar.bz2 bin
echo "✔ Micromamba installed at $MICROMAMBA_DIR/micromamba"


echo "Creating Conda environment at $ENV_DIR…"
JAX_VER="0.4.14"
BASE_PACKAGES=(
  python=3.10 pip pandas matplotlib "numpy<2.0.0"
  biopython scipy pdbfixer seaborn libgfortran5 tqdm
  jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku
  flax="0.9.0" dm-tree joblib ml-collections immutabledict optax
  jax=${JAX_VER} jaxlib=${JAX_VER}
)

if [ -n "$CUDA_VERSION" ]; then
  echo "→ CUDA requested: $CUDA_VERSION"
  # conda-forge currently provides cudatoolkit up to 12.5
  if [ "$CUDA_VERSION" = "12.6" ]; then
    echo "⚠️  cudatoolkit 12.6 not available via conda-forge; falling back to 12.5"
    CTK="12.5"
  else
    CTK="$CUDA_VERSION"
  fi
  CUDA_PACKAGES=( cudatoolkit=${CTK} cuda-nvcc cudnn )
else
  echo "→ No CUDA: CPU‐only install"
  CUDA_PACKAGES=()
fi

ALL_PACKAGES=( "${BASE_PACKAGES[@]}" "${CUDA_PACKAGES[@]}" )

$MICROMAMBA_DIR/micromamba create -y \
  -p $ENV_DIR \
  -c conda-forge \
  -c nvidia \
  --channel https://conda.graylab.jhu.edu \
  "${ALL_PACKAGES[@]}" || { echo "ERROR: Conda env creation failed"; exit 1; }

echo "Verifying JAX import…"
$MICROMAMBA_DIR/micromamba run -p $ENV_DIR python - << 'PYCODE' || exit 1
import jax, jaxlib
print("  JAX:", jax.__version__, "JAXLIB:", jaxlib.__version__)
PYCODE
echo "✔ JAX is working"

echo "Installing ColabDesign…"
$MICROMAMBA_DIR/micromamba run -p $ENV_DIR pip install \
  git+https://github.com/sokrypton/ColabDesign.git --no-deps || exit 1
$MICROMAMBA_DIR/micromamba run -p $ENV_DIR python -c "import colabdesign" || exit 1
echo "✔ ColabDesign installed"

echo "Handling AlphaFold2 weights…"
PARAMS_SYMLINK_DIR="${ENV_DIR}/params"
WEIGHTS_STORAGE_DIR="/tmp/alphafold"
TMP_TAR="/tmp/alphafold_params_2022-12-06.tar"

mkdir -p "$WEIGHTS_STORAGE_DIR" "$PARAMS_SYMLINK_DIR" || exit 1
wget -O "$TMP_TAR" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || exit 1
tar -xvf "$TMP_TAR" -C "$WEIGHTS_STORAGE_DIR"      || exit 1
rm "$TMP_TAR"
for f in "$WEIGHTS_STORAGE_DIR"/*; do
  ln -sf "$f" "$PARAMS_SYMLINK_DIR/"
done
echo "✔ AlphaFold weights symlinked"

echo "Setting executables…"
chmod +x "$(pwd)/functions/dssp"        2>/dev/null || echo "  (dssp missing/OK)"
chmod +x "$(pwd)/functions/DAlphaBall.gcc" 2>/dev/null || echo "  (DAlphaBall.gcc missing/OK)"

echo "Cleaning micromamba cache…"
$MICROMAMBA_DIR/micromamba clean -a -y || echo "  (clean failed)"

t=$SECONDS
echo "✔️ Done! Took $(($t / 3600))h $((($t / 60) % 60))m $(($t % 60))s"
echo "   Env location: $ENV_DIR"
