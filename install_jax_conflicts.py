#!/bin/bash
##########################
# BindCraft Installation #
##########################

# Exit on any error
set -euo pipefail

##################
# Configuration  #
##################
# Directory under /kaggle/working to persist across restarts
MICROMAMBA_DIR="/kaggle/working/micromamba"
ENV_DIR="/kaggle/working/bindcraft_env"
ALPHAFOLD_WEIGHTS_DIR="/kaggle/working/alphafold"
PARAMS_SYMLINK_DIR="${ENV_DIR}/params"
# AlphaFold params archive name
TMP_PARAMS_TAR="/kaggle/working/alphafold_params_2022-12-06.tar"

# CUDA version if provided (e.g. 12.9, 11.8)
CUDA_VERSION=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --cuda)
      CUDA_VERSION="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--cuda <version>]"
      exit 1
      ;;
  esac
done

#################################
# Step 1: Install micromamba    #
#################################
echo "==> Installing micromamba into $MICROMAMBA_DIR..."
mkdir -p "$MICROMAMBA_DIR"
wget -qO micromamba.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest
tar -xvjf micromamba.tar.bz2 -C "$MICROMAMBA_DIR" --strip-components=1 bin/micromamba
chmod +x "$MICROMAMBA_DIR/micromamba"
rm micromamba.tar.bz2
echo "✔ micromamba installed."

#########################################
# Step 2: Create persistent conda env   #
#########################################
echo "==> Creating conda environment at $ENV_DIR..."
"$MICROMAMBA_DIR/micromamba" create -y -p "$ENV_DIR" -c conda-forge -c nvidia \
    python=3.10 pip pandas matplotlib "numpy<2.0.0" biopython scipy pdbfixer tqdm \
    jupyter ffmpeg fsspec py3dmol \
  || { echo "Conda env creation failed"; exit 1; }
echo "✔ Environment created."

#########################################
# Step 3: Install JAX & dependencies    #
#########################################
echo "==> Installing JAX and GPU/CPU backends..."
if [[ -n "$CUDA_VERSION" ]]; then
  MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
  if [[ "$MAJOR" -eq 11 ]]; then
    SUFFIX="cuda11.cudnn86"
  else
    SUFFIX="cuda12.cudnn89"
  fi
  "$MICROMAMBA_DIR/micromamba" run -p "$ENV_DIR" pip install \
    jax==0.4.25 jaxlib==0.4.25+${SUFFIX} \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
  "$MICROMAMBA_DIR/micromamba" run -p "$ENV_DIR" pip install jax==0.4.25 jaxlib==0.4.25
fi

# Pin dependent libraries to versions compatible with JAX 0.4.25
"$MICROMAMBA_DIR/micromamba" run -p "$ENV_DIR" pip install \
  chex==0.1.81 flax==0.7.5 optax==0.1.7 orbax-checkpoint==0.2.4 ml-dtypes==0.4.0

# Verify JAX import
"$MICROMAMBA_DIR/micromamba" run -p "$ENV_DIR" python - <<'PYTEST'
import jax
print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
PYTEST

echo "✔ JAX and dependencies installed."

#########################################
# Step 4: Install ColabDesign          #
#########################################
echo "==> Installing ColabDesign..."
"$MICROMAMBA_DIR/micromamba" run -p "$ENV_DIR" pip install \
  git+https://github.com/sokrypton/ColabDesign.git --no-deps

"$MICROMAMBA_DIR/micromamba" run -p "$ENV_DIR" python - <<'PYTEST'
import colabdesign
print("ColabDesign import successful")
PYTEST

echo "✔ ColabDesign installed."

################################################
# Step 5: Download AlphaFold2 weights & symlink #
################################################
echo "==> Downloading and linking AlphaFold2 weights..."
mkdir -p "$ALPHAFOLD_WEIGHTS_DIR" "$PARAMS_SYMLINK_DIR"
wget -O "$TMP_PARAMS_TAR" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
tar -xvf "$TMP_PARAMS_TAR" -C "$ALPHAFOLD_WEIGHTS_DIR"
rm "$TMP_PARAMS_TAR"
for f in "$ALPHAFOLD_WEIGHTS_DIR"/*; do
  ln -sf "$f" "$PARAMS_SYMLINK_DIR/"
done
echo "✔ AlphaFold2 weights available at $PARAMS_SYMLINK_DIR."

#######################################
# Step 6: Fix binary permissions      #
#######################################
echo "==> Fixing executable permissions..."
chmod +x "$(pwd)/functions/dssp" 2>/dev/null || true
chmod +x "$(pwd)/functions/DAlphaBall.gcc" 2>/dev/null || true

################################
# Step 7: Cleanup micromamba    #
################################
echo "==> Cleaning micromamba cache..."
"$MICROMAMBA_DIR/micromamba" clean -a -y || true

echo "✔ BindCraft install complete in $ENV_DIR"
