#!/bin/bash
################## BindCraft Installation Script (No Conda)
################## Installs directly in the current environment using pip and apt

# Default value for CUDA
cuda=''

# Define the short and long options
OPTIONS=c:
LONGOPTIONS=cuda:

# Parse the command-line options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
if [ $? -ne 0 ]; then
    echo "Failed to parse options." >&2
    exit 1
fi
eval set -- "$PARSED"

# Process the command-line options
while true; do
  case "$1" in
    -c|--cuda)
      cuda="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Invalid option: $1" >&2
      exit 1
      ;;
  esac
done

# Display the parsed options
echo "CUDA version specified: ${cuda:-None}"

############################################################################################################
################## Initialization
SECONDS=0

# Set installation directory
install_dir=$(pwd)

# Check Python installation
if ! command -v python3 &>/dev/null; then
    echo "Error: Python3 is not installed. Please install Python3 and try again."
    exit 1
fi

python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
echo "Using Python version: $python_version"

# Optionally, create and activate a virtual environment
# Uncomment the lines below to use a virtual environment
# echo "Creating a virtual environment..."
# python3 -m venv bindcraft_env || { echo "Error: Failed to create virtual environment."; exit 1; }
# source bindcraft_env/bin/activate || { echo "Error: Failed to activate virtual environment."; exit 1; }
# echo "Virtual environment 'bindcraft_env' activated."

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update || { echo "Error: Failed to update package list."; exit 1; }

# Define required system packages
system_packages=(libgfortran5 ffmpeg wget tar)

# Install system packages
sudo apt-get install -y "${system_packages[@]}" || { echo "Error: Failed to install system packages."; exit 1; }

# Handle CUDA dependencies if specified
if [ -n "$cuda" ]; then
    echo "CUDA specified: $cuda. Installing CUDA and cuDNN dependencies..."

    # Example for CUDA 11.8; adjust as needed
    sudo apt-get install -y nvidia-cuda-toolkit || { echo "Error: Failed to install CUDA toolkit."; exit 1; }
    
    # Install cuDNN (requires NVIDIA account and manual download)
    # Uncomment and modify the lines below based on your CUDA version and cuDNN availability
    # echo "Installing cuDNN..."
    # tar -xzvf cudnn-11.8-linux-x64-v8.6.0.163.tgz
    # sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
    # sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64/
    # sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip || { echo "Error: Failed to upgrade pip."; exit 1; }

# Install required Python packages
echo "Installing Python packages..."
python_packages=(
    pip
    pandas
    matplotlib
    "numpy<2.0.0"
    biopython
    scipy
    pdbfixer
    seaborn
    tqdm
    jupyter
    ffmpeg-python
    pyrosetta
    fsspec
    py3dmol
    chex
    dm-haiku
    "flax<0.10.0"
    dm-tree
    joblib
    ml-collections
    immutabledict
    optax
    jaxlib
    jax
)

# Adjust jaxlib for CUDA if specified
if [ -n "$cuda" ]; then
    echo "Installing JAX with CUDA support..."
    # Determine the correct jaxlib version based on CUDA
    # Example for CUDA 11.8
    python_packages+=("jaxlib==0.4.13+cuda118" "jax==0.4.13") # Replace with appropriate versions
    # Note: You might need to specify the exact URL for the CUDA-enabled jaxlib wheel
else
    echo "Installing JAX without CUDA support..."
    python_packages+=("jaxlib" "jax")
fi

# Install all Python packages
python3 -m pip install "${python_packages[@]}" || { echo "Error: Failed to install Python packages."; exit 1; }

# Verify installation of Python packages
echo "Verifying installed Python packages..."
required_packages=(pip pandas matplotlib numpy biopython scipy pdbfixer seaborn tqdm jupyter ffmpeg-python pyrosetta fsspec py3dmol chex dm-haiku flax dm-tree joblib ml-collections immutabledict optax jaxlib jax)

missing_packages=()
for pkg in "${required_packages[@]}"; do
    python3 -c "import $pkg" &>/dev/null
    if [ $? -ne 0 ]; then
        missing_packages+=("$pkg")
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "Error: The following Python packages are missing:"
    for pkg in "${missing_packages[@]}"; do
        echo " - $pkg"
    done
    exit 1
fi

# Install ColabDesign
echo "Installing ColabDesign..."
pip3 install git+https://github.com/sokrypton/ColabDesign.git --no-deps || { echo "Error: Failed to install ColabDesign."; exit 1; }

# Verify ColabDesign installation
python3 -c "import colabdesign" &>/dev/null || { echo "Error: colabdesign module not found after installation."; exit 1; }

# Cleanup pip cache to save space
echo "Cleaning up pip cache..."
pip3 cache purge || { echo "Warning: Failed to purge pip cache."; }

# Download AlphaFold2 model weights
echo "Downloading AlphaFold2 model weights..."
params_dir="${install_dir}/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"

mkdir -p "${params_dir}" || { echo "Error: Failed to create weights directory."; exit 1; }
wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { echo "Error: Failed to download AlphaFold2 weights."; exit 1; }
[ -s "${params_file}" ] || { echo "Error: Downloaded AlphaFold2 weights file is empty."; exit 1; }

# Extract AlphaFold2 weights
echo "Extracting AlphaFold2 weights..."
tar -xvf "${params_file}" -C "${params_dir}" || { echo "Error: Failed to extract AlphaFold2 weights."; exit 1; }
[ -f "${params_dir}/params_model_5_ptm.npz" ] || { echo "Error: Extracted AlphaFold2 weights not found."; exit 1; }
rm "${params_file}" || { echo "Warning: Failed to remove the downloaded weights archive."; }

# Set executable permissions
echo "Setting executable permissions..."
chmod +x "${install_dir}/functions/dssp" || { echo "Error: Failed to set execute permission for dssp."; exit 1; }
chmod +x "${install_dir}/functions/DAlphaBall.gcc" || { echo "Error: Failed to set execute permission for DAlphaBall.gcc."; exit 1; }

# Cleanup pip cache to save space
echo "Cleaning up pip cache..."
pip3 cache purge || { echo "Warning: Failed to purge pip cache."; }

# Finish installation
echo "BindCraft environment setup complete!"

############################################################################################################
################## Installation Summary
t=$SECONDS 
echo "----------------------------------------"
echo "Successfully finished BindCraft installation!"
echo "Installation took $((t / 3600)) hours, $(((t / 60) % 60)) minutes and $((t % 60)) seconds."
echo "To activate the environment (if using a virtual environment), run:"
echo "  source bindcraft_env/bin/activate"
echo "----------------------------------------"
