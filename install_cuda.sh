#!/bin/bash
################## BindCraft Installation Script for Kaggle (Modified Version with AlphaFold weights symlinked from /tmp)

# This script installs BindCraft dependencies into a prefix-based Micromamba environment,
# with conditional support for CUDA-enabled GPU acceleration.
# It downloads AlphaFold2 weights to /tmp/alphafold and creates symlinks in /kaggle/working.

################## Configuration

CUDA_VERSION=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: $0 [--cuda <version>]"
            exit 1
            ;;
    esac
done

MICROMAMBA_DIR="/tmp/micromamba"
ENV_DIR="/kaggle/working/bindcraft_env"

################## Step 1: Install Micromamba
echo "Installing Micromamba..."
wget -qO micromamba.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest || exit 1
tar -xvjf micromamba.tar.bz2 bin/micromamba || exit 1
chmod +x bin/micromamba || exit 1
mkdir -p $MICROMAMBA_DIR || exit 1
mv bin/micromamba $MICROMAMBA_DIR/micromamba || exit 1
rm -rf micromamba.tar.bz2 bin
echo "Micromamba installed at $MICROMAMBA_DIR/micromamba"

################## Step 2: Create Conda Environment
echo "Creating Conda environment at $ENV_DIR..."
BASE_PACKAGES=(
    python=3.10 pip pandas matplotlib "numpy<2.0.0" biopython scipy pdbfixer
    seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex
    dm-haiku flax="0.9.0" dm-tree joblib ml-collections immutabledict optax
)

if [ -n "$CUDA_VERSION" ]; then
    echo "CUDA version specified: $CUDA_VERSION"
    CUDA_PACKAGES=(cuda-nvcc cudnn)
else
    echo "No CUDA version specified. CPU-only installation."
    CUDA_PACKAGES=()
fi

ALL_PACKAGES=("${BASE_PACKAGES[@]}" "${CUDA_PACKAGES[@]}")

$MICROMAMBA_DIR/micromamba create -y \
    -p $ENV_DIR \
    -c conda-forge -c nvidia \
    --channel https://conda.graylab.jhu.edu \
    "${ALL_PACKAGES[@]}" || exit 1

################## Step 3: Install JAX via pip
echo "Installing JAX..."
if [ -n "$CUDA_VERSION" ]; then
    CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f1)
    $MICROMAMBA_DIR/micromamba run -p $ENV_DIR pip install --upgrade "jax[cuda${CUDA_MAJOR_VERSION}]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || exit 1
else
    $MICROMAMBA_DIR/micromamba run -p $ENV_DIR pip install --upgrade jax jaxlib || exit 1
fi

$MICROMAMBA_DIR/micromamba run -p $ENV_DIR python -c "import jax" || exit 1
echo "JAX installed."

################## Step 4: Install ColabDesign
echo "Installing ColabDesign..."
$MICROMAMBA_DIR/micromamba run -p $ENV_DIR pip install git+https://github.com/sokrypton/ColabDesign.git --no-deps || exit 1
$MICROMAMBA_DIR/micromamba run -p $ENV_DIR python -c "import colabdesign" || exit 1
echo "ColabDesign installed."

################## Step 5: Download AlphaFold2 Weights and Create Symlinks
echo "Handling AlphaFold2 weights..."
PARAMS_SYMLINK_DIR="${ENV_DIR}/params"
WEIGHTS_STORAGE_DIR="/tmp/alphafold"
TMP_PARAMS_TAR="/tmp/alphafold_params_2022-12-06.tar"

mkdir -p "$WEIGHTS_STORAGE_DIR" "$PARAMS_SYMLINK_DIR" || exit 1

echo "Downloading AlphaFold weights to $TMP_PARAMS_TAR..."
wget -O "$TMP_PARAMS_TAR" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || exit 1

echo "Extracting AlphaFold weights to $WEIGHTS_STORAGE_DIR..."
tar -xvf "$TMP_PARAMS_TAR" -C "$WEIGHTS_STORAGE_DIR" || exit 1
rm "$TMP_PARAMS_TAR" || echo "Warning: Failed to delete tarball."

echo "Creating symlinks in $PARAMS_SYMLINK_DIR..."
for file in "$WEIGHTS_STORAGE_DIR"/*; do
    ln -sf "$file" "$PARAMS_SYMLINK_DIR/"
done
echo "AlphaFold weights symlinked."

################## Step 6: Adjust Permissions
echo "Setting executable permissions..."
chmod +x "$(pwd)/functions/dssp" 2>/dev/null || echo "dssp not found or already executable"
chmod +x "$(pwd)/functions/DAlphaBall.gcc" 2>/dev/null || echo "DAlphaBall.gcc not found or already executable"

################## Step 7: Clean Up
echo "Cleaning Micromamba cache..."
$MICROMAMBA_DIR/micromamba clean -a -y || echo "Warning: micromamba clean failed."

################## Done
t=$SECONDS
echo "✔️ BindCraft installation complete!"
echo "⏱️ Took $(($t / 3600))h $((($t / 60) % 60))m $(($t % 60))s"
echo "📦 Environment installed in $ENV_DIR"
