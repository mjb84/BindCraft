#!/bin/bash
############################################################################################################
################## BindCraft Pip-Focused Installation Script ###############################################
################## This script uses a hybrid pip/conda approach to avoid dependency conflicts ##############
############################################################################################################

# Default value for pkg_manager
pkg_manager='conda'

# --- Argument Parsing for --pkg_manager (conda or mamba) ---
OPTIONS=p:
LONGOPTIONS=pkg_manager:

PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

while true; do
  case "$1" in
    -p|--pkg_manager)
      pkg_manager="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo -e "Invalid option $1. Use -p or --pkg_manager ['conda' or 'mamba']" >&2
      exit 1
      ;;
  esac
done

echo -e "--- Using package manager: $pkg_manager ---"

################## Initialisation ##################
SECONDS=0
install_dir=$(pwd)
CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
echo -e "Conda is installed at: $CONDA_BASE"

################## Environment Creation ##################
echo -e "\n--- Step 1: Creating a clean Conda environment for BindCraft ---"
$pkg_manager create --name BindCraft python=3.10 pip -y || { echo -e "Error: Failed to create BindCraft conda environment"; exit 1; }

# Activate the new environment
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/BindCraft || { echo -e "Error: Failed to activate the BindCraft environment."; exit 1; }
echo -e "BindCraft environment activated at ${CONDA_BASE}/envs/BindCraft"

################## JAX and CUDA Installation (Pip) ##################
echo -e "\n--- Step 2: Installing JAX with CUDA support using pip ---"
pip install -U "jax[cuda12]" || { echo -e "Error: Failed to install JAX with CUDA support."; exit 1; }

# --- CRUCIAL VERIFICATION STEP ---
echo -e "\n--- Step 3: Verifying JAX can detect the GPU ---"
if python -c "import jax; print(jax.devices())" | grep -q 'CudaDevice'; then
    echo -e "✅ JAX successfully detected the GPU."
else
    echo -e "❌ Error: JAX could not find a CUDA-enabled GPU. Installation cannot proceed."
    exit 1
fi

################## Conda-Only Dependencies ##################
echo -e "\n--- Step 4: Installing Conda-specific packages (pdbfixer, pyrosetta) ---"
$pkg_manager install pdbfixer pyrosetta -c conda-forge --channel https://conda.graylab.jhu.edu -y || { echo -e "Error: Failed to install conda-specific packages."; exit 1; }


################## Remaining Dependencies (Pip) ##################
echo -e "\n--- Step 5: Installing remaining Python packages with pip ---"
pip install pandas matplotlib numpy"<2.0.0" biopython scipy seaborn tqdm jupyter ffmpeg fsspec py3dmol chex dm-haiku flax"<0.10.0" dm-tree joblib ml-collections immutabledict optax PyYAML tabulate typing-extensions absl-py pytz python-dateutil tzdata contourpy cycler fonttools kiwisolver packaging pillow pyparsing || { echo -e "Error: Failed to install remaining pip packages."; exit 1; }

################## ColabDesign Installation ##################
echo -e "\n--- Step 6: Installing ColabDesign ---"
pip install git+https://github.com/sokrypton/ColabDesign.git --no-deps || { echo -e "Error: Failed to install ColabDesign"; exit 1; }

################## AlphaFold2 Weights Download ##################
echo -e "\n--- Step 7: Downloading and extracting AlphaFold2 model weights ---"
params_dir="${install_dir}/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"

mkdir -p "${params_dir}" || { echo -e "Error: Failed to create weights directory"; exit 1; }
if [ ! -f "${params_dir}/params_model_5_ptm.npz" ]; then
    wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { echo -e "Error: Failed to download AlphaFold2 weights"; exit 1; }
    tar -xf "${params_file}" -C "${params_dir}" || { echo -e "Error: Failed to extract AlphaFold2 weights"; exit 1; }
    rm "${params_file}" || { echo -e "Warning: Failed to remove AlphaFold2 weights archive"; }
else
    echo -e "AlphaFold2 weights already exist. Skipping download."
fi

################## Final Setup ##################
echo -e "\n--- Step 8: Setting permissions for executables ---"
chmod +x "${install_dir}/functions/dssp" || { echo -e "Error: Failed to chmod dssp"; exit 1; }
chmod +x "${install_dir}/functions/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

# Deactivate for a clean exit
conda deactivate

################## Cleanup and Finish ##################
echo -e "\n--- Cleaning up temporary files ---"
$pkg_manager clean -a -y

t=$SECONDS
echo -e "\n✅ Successfully finished BindCraft installation!"
echo -e "Activate the environment using the command: \"conda activate BindCraft\""
echo -e "\nInstallation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."
