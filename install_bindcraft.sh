#!/bin/bash
################## BindCraft Installation Script for Kaggle (Option 3)
# This script installs BindCraft dependencies directly into Kaggle's system environment (/usr/local)
# using Conda without activating a separate environment.

################## Configuration
# Default values for package manager and CUDA
pkg_manager='conda'
cuda=''

# Define the short and long options for script arguments
OPTIONS=p:c:
LONGOPTIONS=pkg_manager:,cuda:

# Parse the command-line options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
if [ $? -ne 0 ]; then
    echo -e "Error: Failed to parse options." >&2
    exit 1
fi
eval set -- "$PARSED"

# Process the command-line options
while true; do
  case "$1" in
    -p|--pkg_manager)
      pkg_manager="$2"
      shift 2
      ;;
    -c|--cuda)
      cuda="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo -e "Error: Invalid option $1" >&2
      exit 1
      ;;
  esac
done

# Display the selected configuration
echo -e "Package Manager: $pkg_manager"
echo -e "CUDA Version: $cuda"

############################################################################################################
################## Initialisation
SECONDS=0

# 1. Install Miniconda Locally
CONDA_INSTALL_DIR="/kaggle/working/miniconda"
if [ ! -d "$CONDA_INSTALL_DIR" ]; then
    echo -e "\nInstalling Miniconda..."
    wget -q -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        || { echo -e "Error: Failed to download Miniconda installer."; exit 1; }
    bash miniconda.sh -b -p "$CONDA_INSTALL_DIR" \
        || { echo -e "Error: Failed to install Miniconda."; exit 1; }
    rm miniconda.sh
    echo -e "Miniconda installed at $CONDA_INSTALL_DIR"
else
    echo -e "\nMiniconda already installed at $CONDA_INSTALL_DIR"
fi

# 2. Update PATH to Include Conda
export PATH="$CONDA_INSTALL_DIR/bin:$PATH"
echo -e "Conda added to PATH."

# 3. Update Conda Itself
echo -e "\nUpdating Conda..."
conda update conda -y \
    || { echo -e "Error: Failed to update Conda."; exit 1; }

# 4. Verify Conda Installation
CONDA_BASE=$(conda info --base 2>/dev/null) \
    || { echo -e "Error: Conda is not installed or cannot be initialised."; exit 1; }
echo -e "Conda is installed at: $CONDA_BASE"

# 5. Install libarchive into /usr/local
echo -e "\nInstalling libarchive into /usr/local..."
conda install -y -p /usr/local -c conda-forge libarchive=3.6.2 \
    || { echo -e "Error: Failed to install libarchive."; exit 1; }

# 6. Install BindCraft Dependencies into /usr/local
echo -e "\nInstalling BindCraft dependencies into /usr/local..."
if [ -n "$cuda" ]; then
    echo -e "Installing with CUDA support: $cuda"
    CONDA_OVERRIDE_CUDA="$cuda" conda install -y -p /usr/local \
        -c conda-forge -c nvidia --channel https://conda.graylab.jhu.edu \
        pip pandas matplotlib "numpy<2.0.0" biopython scipy pdbfixer seaborn libgfortran5 \
        tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku "flax<0.10.0" \
        dm-tree joblib ml-collections immutabledict optax jaxlib="*=*cuda*" jax \
        cuda-nvcc cudnn \
        || { echo -e "Error: Failed to install Conda packages with CUDA."; exit 1; }
else
    echo -e "Installing without CUDA support."
    conda install -y -p /usr/local \
        -c conda-forge -c nvidia --channel https://conda.graylab.jhu.edu \
        pip pandas matplotlib "numpy<2.0.0" biopython scipy pdbfixer seaborn libgfortran5 \
        tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku "flax<0.10.0" \
        dm-tree joblib ml-collections immutabledict optax jaxlib jax cuda-nvcc cudnn \
        || { echo -e "Error: Failed to install Conda packages."; exit 1; }
fi

# 7. Verify Installation of Required Packages
echo -e "\nVerifying installation of required packages..."
required_packages=(pip pandas libgfortran5 matplotlib numpy biopython scipy pdbfixer \
    seaborn tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku dm-tree \
    joblib ml-collections immutabledict optax jaxlib jax cuda-nvcc cudnn)
missing_packages=()

for pkg in "${required_packages[@]}"; do
    conda list -p /usr/local "$pkg" >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        missing_packages+=("$pkg")
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "Error: The following packages are missing from /usr/local environment:"
    for pkg in "${missing_packages[@]}"; do
        echo -e " - $pkg"
    done
    exit 1
else
    echo -e "All required packages are successfully installed in /usr/local."
fi

# 8. Install ColabDesign via pip
echo -e "\nInstalling ColabDesign..."
pip3 install git+https://github.com/sokrypton/ColabDesign.git --no-deps \
    || { echo -e "Error: Failed to install ColabDesign."; exit 1; }
python -c "import colabdesign" >/dev/null 2>&1 \
    || { echo -e "Error: colabdesign module not found after installation."; exit 1; }
echo -e "ColabDesign successfully installed."

# 9. Download AlphaFold2 Model Weights
echo -e "\nDownloading AlphaFold2 model weights..."
params_dir="$(pwd)/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"

mkdir -p "${params_dir}" \
    || { echo -e "Error: Failed to create weights directory."; exit 1; }
wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" \
    || { echo -e "Error: Failed to download AlphaFold2 weights."; exit 1; }
[ -s "${params_file}" ] \
    || { echo -e "Error: Downloaded AlphaFold2 weights file is empty."; exit 1; }

# 10. Extract AlphaFold2 Weights
echo -e "\nExtracting AlphaFold2 weights..."
tar tf "${params_file}" >/dev/null 2>&1 \
    || { echo -e "Error: Corrupt AlphaFold2 weights archive."; exit 1; }
tar -xvf "${params_file}" -C "${params_dir}" \
    || { echo -e "Error: Failed to extract AlphaFold2 weights."; exit 1; }
[ -f "${params_dir}/params_model_5_ptm.npz" ] \
    || { echo -e "Error: Extracted AlphaFold2 weights file not found."; exit 1; }
rm "${params_file}" \
    || { echo -e "Warning: Failed to remove AlphaFold2 weights archive."; }

# 11. Change Permissions for Executables
echo -e "\nChanging permissions for executables..."
chmod +x "$(pwd)/functions/dssp" \
    || { echo -e "Error: Failed to chmod dssp."; exit 1; }
chmod +x "$(pwd)/functions/DAlphaBall.gcc" \
    || { echo -e "Error: Failed to chmod DAlphaBall.gcc."; exit 1; }

# 12. Clean Up Conda Temporary Files
echo -e "\nCleaning up Conda temporary files to save space..."
conda clean -a -y \
    || { echo -e "Warning: Failed to clean Conda cache."; }
echo -e "Conda temporary files cleaned up."

# 13. Finalization
t=$SECONDS
echo -e "\nSuccessfully finished BindCraft installation!"
echo -e "\nInstallation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes, and $(($t % 60)) seconds."
echo -e "\nAll dependencies are installed in /usr/local and ready to use."
