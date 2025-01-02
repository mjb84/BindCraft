#!/bin/bash
################## BindCraft Installation Script for Kaggle (Option 2)
# This script installs BindCraft dependencies into a prefix-based Micromamba environment.

################## Configuration
# Define environment directories
MICROMAMBA_DIR="/tmp/micromamba"
ENV_DIR="/kaggle/working/bindcraft_env"

# CUDA version (if needed)
CUDA_VERSION="12.6"  # e.g., "11.2" or leave empty for no CUDA support

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

################## Step 2: Create Conda Environment
echo "Creating Conda environment at $ENV_DIR..."
$MICROMAMBA_DIR/micromamba create -y \
    -p $ENV_DIR \
    -c conda-forge \
    -c nvidia \
    --channel https://conda.graylab.jhu.edu \
    python=3.10 \
    pip \
    pandas \
    matplotlib \
    "numpy<2.0.0" \
    biopython \
    scipy \
    pdbfixer \
    seaborn \
    libgfortran5 \
    tqdm \
    jupyter \
    ffmpeg \
    pyrosetta \
    fsspec \
    py3dmol \
    chex \
    dm-haiku \
    flax="0.9.0" \
    dm-tree \
    joblib \
    ml-collections \
    immutabledict \
    optax \
    jaxlib \
    jax \
    cuda-nvcc \
    cudnn \
    || { echo "Error: Failed to create Conda environment."; exit 1; }

################## Step 3: Install ColabDesign via pip
echo "Installing ColabDesign..."
$MICROMAMBA_DIR/micromamba run -p $ENV_DIR pip install git+https://github.com/sokrypton/ColabDesign.git --no-deps \
    || { echo "Error: Failed to install ColabDesign."; exit 1; }

# Verify ColabDesign installation
$MICROMAMBA_DIR/micromamba run -p $ENV_DIR python -c "import colabdesign" \
    || { echo "Error: colabdesign module not found after installation."; exit 1; }
echo "ColabDesign successfully installed."

################## Step 4: Download and Extract AlphaFold2 Weights
echo "Downloading AlphaFold2 model weights..."
PARAMS_DIR="${ENV_DIR}/params"
PARAMS_FILE="${PARAMS_DIR}/alphafold_params_2022-12-06.tar"
mkdir -p $PARAMS_DIR \
    || { echo "Error: Failed to create parameters directory."; exit 1; }
wget -O $PARAMS_FILE "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" \
    || { echo "Error: Failed to download AlphaFold2 weights."; exit 1; }
tar -xvf $PARAMS_FILE -C $PARAMS_DIR \
    || { echo "Error: Failed to extract AlphaFold2 weights."; exit 1; }
rm $PARAMS_FILE \
    || { echo "Warning: Failed to remove AlphaFold2 weights archive."; }
echo "AlphaFold2 weights downloaded and extracted to $PARAMS_DIR."

################## Step 5: Adjust Permissions for Executables
echo "Changing permissions for executables..."
chmod +x "$(pwd)/functions/dssp" \
    || { echo "Error: Failed to chmod dssp."; exit 1; }
chmod +x "$(pwd)/functions/DAlphaBall.gcc" \
    || { echo "Error: Failed to chmod DAlphaBall.gcc."; exit 1; }
echo "Permissions updated."

################## Step 6: Clean Up Micromamba Cache
echo "Cleaning up Micromamba cache..."
$MICROMAMBA_DIR/micromamba clean -a -y \
    || { echo "Warning: Failed to clean Micromamba cache."; }
echo "Micromamba cache cleaned."

################## Finalization
t=$SECONDS
echo "Successfully finished BindCraft installation!"
echo "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."
echo "All dependencies are installed in $ENV_DIR and ready to use."
