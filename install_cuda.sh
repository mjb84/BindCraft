#!/bin/bash
################## BindCraft Installation Script for Kaggle (Adapted Version)
# This script installs BindCraft dependencies into a prefix-based Micromamba environment,
# with conditional support for CUDA-enabled GPU acceleration.

################## Configuration

# Default CUDA version (set to empty for CPU-only installation)
CUDA_VERSION=""

# Parse command-line arguments for CUDA version
# Usage: ./install_bindcraft.sh --cuda 12.6
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

# Define environment directories
MICROMAMBA_DIR="/tmp/micromamba"
ENV_DIR="/kaggle/working/bindcraft_env"

################## Step 1: Install Micromamba
echo "Installing Micromamba..."
wget -qO micromamba.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest \
    || { echo "Error: Failed to download Micromamba."; exit 1; }
tar -xvjf micromamba.tar.bz2 bin/micromamba \
    || { echo "Error: Failed to extract Micromamba."; exit 1; }
chmod +x bin/micromamba \
    || { echo "Error: Failed to make Micromamba executable."; exit 1; }
mkdir -p $MICROMAMBA_DIR \
    || { echo "Error: Failed to create Micromamba directory."; exit 1; }
mv bin/micromamba $MICROMAMBA_DIR/micromamba \
    || { echo "Error: Failed to move Micromamba."; exit 1; }
rm -rf micromamba.tar.bz2 bin
echo "Micromamba installed at $MICROMAMBA_DIR/micromamba"

################## Step 2: Create Conda Environment with Conditional CUDA Support
echo "Creating Conda environment at $ENV_DIR..."

# Define base packages (excluding jax and jaxlib)
BASE_PACKAGES=(
    python=3.10
    pip
    pandas
    matplotlib
    "numpy<2.0.0"
    biopython
    scipy
    pdbfixer
    seaborn
    libgfortran5
    tqdm
    jupyter
    ffmpeg
    pyrosetta
    fsspec
    py3dmol
    chex
    dm-haiku
    flax="0.9.0"
    dm-tree
    joblib
    ml-collections
    immutabledict
    optax
)

# Define CUDA-specific packages (excluding jax and jaxlib)
if [ -n "$CUDA_VERSION" ]; then
    echo "CUDA version specified: $CUDA_VERSION"
    CUDA_PACKAGES=(
        cuda-nvcc
        cudnn
        # Add any other CUDA-specific packages here if needed
    )
else
    echo "No CUDA version specified. Proceeding with CPU-only installation."
    CUDA_PACKAGES=()
fi

# Combine base and CUDA packages
ALL_PACKAGES=("${BASE_PACKAGES[@]}" "${CUDA_PACKAGES[@]}")

# Install packages using Micromamba
$MICROMAMBA_DIR/micromamba create -y \
    -p $ENV_DIR \
    -c conda-forge \
    -c nvidia \
    --channel https://conda.graylab.jhu.edu \
    "${ALL_PACKAGES[@]}" \
    || { echo "Error: Failed to create Conda environment."; exit 1; }

################## Step 3: Install JAX and JAXLIB via pip
echo "Installing JAX and JAXLIB via pip..."

if [ -n "$CUDA_VERSION" ]; then
    # Extract major CUDA version (e.g., 12 from 12.6)
    CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d. -f1)
    echo "Installing JAX with CUDA support (CUDA $CUDA_MAJOR_VERSION)..."
    $MICROMAMBA_DIR/micromamba run -p $ENV_DIR pip install --upgrade "jax[cuda${CUDA_MAJOR_VERSION}]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
        || { echo "Error: Failed to install JAX with CUDA support via pip."; exit 1; }
else
    echo "Installing JAX (CPU-only)..."
    $MICROMAMBA_DIR/micromamba run -p $ENV_DIR pip install --upgrade jax jaxlib \
        || { echo "Error: Failed to install JAX via pip."; exit 1; }
fi

# Verify JAX installation
$MICROMAMBA_DIR/micromamba run -p $ENV_DIR python -c "import jax" \
    || { echo "Error: jax module not found after installation."; exit 1; }
echo "JAX successfully installed."

################## Step 4: Install ColabDesign via pip
echo "Installing ColabDesign..."
$MICROMAMBA_DIR/micromamba run -p $ENV_DIR pip install git+https://github.com/sokrypton/ColabDesign.git --no-deps \
    || { echo "Error: Failed to install ColabDesign."; exit 1; }

# Verify ColabDesign installation
$MICROMAMBA_DIR/micromamba run -p $ENV_DIR python -c "import colabdesign" \
    || { echo "Error: colabdesign module not found after installation."; exit 1; }
echo "ColabDesign successfully installed."

################## Step 7: Clean Up Micromamba Cache
echo "Cleaning up Micromamba cache..."
$MICROMAMBA_DIR/micromamba clean -a -y \
    || { echo "Warning: Failed to clean Micromamba cache."; }
echo "Micromamba cache cleaned."

################## Step 5: Download and Extract AlphaFold2 Weights
echo "Downloading AlphaFold2 model weights..."
PARAMS_DIR="${ENV_DIR}/params"
TMP_DOWNLOAD_DIR="/tmp"
TMP_PARAMS_FILE="${TMP_DOWNLOAD_DIR}/alphafold_params_2022-12-06.tar"
FINAL_PARAMS_FILE="${PARAMS_DIR}/alphafold_params_2022-12-06.tar"

# Create the parameters directory
mkdir -p $PARAMS_DIR \
    || { echo "Error: Failed to create parameters directory."; exit 1; }

# Download AF2 weights to /tmp
echo "Downloading AF2 weights to $TMP_PARAMS_FILE..."
wget -O $TMP_PARAMS_FILE "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" \
    || { echo "Error: Failed to download AlphaFold2 weights."; exit 1; }

# Verify the download
if [ ! -s "$TMP_PARAMS_FILE" ]; then
    echo "Error: Downloaded AlphaFold2 weights file is empty or does not exist."
    exit 1
fi

# Extract AF2 weights to PARAMS_DIR
echo "Extracting AF2 weights to $PARAMS_DIR..."
tar -xvf $TMP_PARAMS_FILE -C $PARAMS_DIR \
    || { echo "Error: Failed to extract AlphaFold2 weights."; exit 1; }

# Remove the downloaded tar file from /tmp
echo "Removing downloaded AF2 weights tar file from /tmp..."
rm $TMP_PARAMS_FILE \
    || { echo "Warning: Failed to remove AlphaFold2 weights archive from /tmp."; }

echo "AlphaFold2 weights downloaded and extracted to $PARAMS_DIR."

################## Step 6: Adjust Permissions for Executables
echo "Changing permissions for executables..."
chmod +x "$(pwd)/functions/dssp" \
    || { echo "Error: Failed to chmod dssp."; exit 1; }
chmod +x "$(pwd)/functions/DAlphaBall.gcc" \
    || { echo "Error: Failed to chmod DAlphaBall.gcc."; exit 1; }
echo "Permissions updated."



################## Finalization
t=$SECONDS
echo "Successfully finished BindCraft installation!"
echo "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."
echo "All dependencies are installed in $ENV_DIR and ready to use."
