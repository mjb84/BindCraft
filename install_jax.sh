#!/usr/bin/env bash
##############################################################################
# BindCraft – Kaggle-compatible installer
# Author: <your-name>          Date: 2025-05-27
# Tested on: Kaggle Python 3.10 GPU image (CUDA 11.8) – May-2025
##############################################################################
set -euo pipefail
IFS=$'\n\t'

############################## command-line flags ############################
pkg_manager=''      # autodetect (conda/mamba/micromamba) unless overridden
cuda=''             # explicit CUDA toolkit version, "--cpu" for CPU-only

OPTIONS=p:c:
LONGOPTIONS=pkg_manager:,cuda:,cpu

PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

while true; do
  case "$1" in
    -p|--pkg_manager) pkg_manager="$2"; shift 2 ;;
    -c|--cuda)        cuda="$2";       shift 2 ;;
    --cpu)            cuda="";         shift   ;;
    --) shift; break ;;
    *) echo "❌  Unknown option $1"; exit 1 ;;
  esac
done

############################## environment probe #############################
KAGGLE_ENV=false
[ -d /kaggle ] && KAGGLE_ENV=true

# Resolve which package manager we can use (conda → mamba → micromamba)
_find_pm() {
    command -v conda      >/dev/null 2>&1 && { echo conda;      return; }
    command -v mamba      >/dev/null 2>&1 && { echo mamba;      return; }
    command -v micromamba >/dev/null 2>&1 && { echo micromamba; return; }
}

if [[ -z "$pkg_manager" ]]; then
    pkg_manager=$(_find_pm || true)
fi

# If nothing exists (typical on Kaggle), bootstrap micromamba into working dir
if [[ -z "$pkg_manager" || ( "$pkg_manager" == "micromamba" && ! command -v micromamba >/dev/null ) ]]; then
    echo "ℹ️  Boot-strapping standalone micromamba ..."
    INSTALL_ROOT=/kaggle/working/micromamba
    mkdir -p "$INSTALL_ROOT"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
        tar -xvj -C "$INSTALL_ROOT" bin/micromamba
    export PATH="$INSTALL_ROOT/bin:$PATH"
    pkg_manager=micromamba
fi

echo "▶ Using package manager: $pkg_manager"

############################## base prefix setup #############################
# Micromamba does not require/ship a base env, so we invent one in /kaggle/working
if [[ "$pkg_manager" == "micromamba" ]]; then
    export MAMBA_ROOT_PREFIX=/kaggle/working/conda_root
    mkdir -p "$MAMBA_ROOT_PREFIX"
fi

# Determine base path for activation later
CONDA_BASE=$(
    case "$pkg_manager" in
      conda|mamba)      "$pkg_manager" info --base ;;
      micromamba)       echo "$MAMBA_ROOT_PREFIX" ;;
    esac
)
echo "▶ Conda-style root: $CONDA_BASE"

############################## shell integration #############################
# Ensure current shell knows about activation commands without spawning subshells
eval "$("$pkg_manager" shell hook -s bash)"

############################## create environment ############################
echo "▶ Creating environment BindCraft (python 3.10)"
"$pkg_manager" create -y -n BindCraft python=3.10

# Activate once for the remainder of the script
"$pkg_manager" activate BindCraft
if [[ "$CONDA_DEFAULT_ENV" != "BindCraft" ]]; then
    echo "❌  Failed to activate BindCraft environment"; exit 1
fi

############################## package install ################################
echo "▶ Installing conda packages (${cuda:+GPU-enabled CUDA=$cuda})"

# Channels as arrays
BASE_CHAN=( -c conda-forge --channel https://conda.graylab.jhu.edu )
GPU_CHAN=( -c nvidia )

# Base list as array
COMMON_PKGS=(
  pip pandas matplotlib "numpy<2.0.0" biopython scipy pdbfixer
  seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta
  fsspec py3dmol chex dm-haiku "flax<0.10.0" dm-tree joblib
  ml-collections immutabledict optax
)

if [[ -n "$cuda" ]]; then
    GPU_PKGS=( jaxlib=*=*cuda* jax cuda-nvcc cudnn )
    "$pkg_manager" install -y \
      "${COMMON_PKGS[@]}" "${GPU_PKGS[@]}" "${BASE_CHAN[@]}" "${GPU_CHAN[@]}"
else
    "$pkg_manager" install -y \
      "${COMMON_PKGS[@]}" jaxlib jax "${BASE_CHAN[@]}"
fi

############################## integrity check ################################
echo "▶ Verifying core packages"
required_packages=(pip pandas matplotlib numpy scipy jaxlib)

missing=()
for pkg in "${required_packages[@]}"; do
    "$pkg_manager" list "$pkg" | grep -qE "^$pkg " || missing+=("$pkg")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "❌  Package(s) missing: ${missing[*]}"; exit 1
fi

############################## pip installs ###################################
echo "▶ Installing ColabDesign via pip (no deps)"
pip install --no-deps git+https://github.com/sokrypton/ColabDesign.git
python -c "import colabdesign" || { echo "❌  ColabDesign import failed"; exit 1; }

############################## AlphaFold weights ##############################
echo "▶ Downloading AlphaFold2 parameters"
PARAM_DIR=/tmp/alphafold_params
mkdir -p "$PARAM_DIR"
AF_ARCHIVE="$PARAM_DIR/alphafold_params_2022-12-06.tar"

wget -q --show-progress -O "$AF_ARCHIVE" \
     https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf "$AF_ARCHIVE" -C "$PARAM_DIR"
rm "$AF_ARCHIVE"

# Sanity check
[[ -f "$PARAM_DIR/params_model_5_ptm.npz" ]] ||
    { echo "❌  AlphaFold params missing"; exit 1; }

############################## executable perms ################################
chmod +x "$(pwd)/functions/dssp"          || echo "⚠️  dssp chmod skipped"
chmod +x "$(pwd)/functions/DAlphaBall.gcc" || echo "⚠️  DAlphaBall chmod skipped"

############################## clean-up #######################################
"$pkg_manager" clean -a -y
echo "🧹  Package cache cleaned"

############################## summary ########################################
SECONDS=$((SECONDS))
printf "\n✅  BindCraft environment ready.\n"
printf "▶ Activate anytime inside this notebook with:\n"
printf "   eval \"\$($pkg_manager shell hook -s bash)\" && $pkg_manager activate BindCraft\n"
printf "⏱  Installation time: %d h %d min %d s\n" \
        "$((SECONDS/3600))" "$(((SECONDS/60)%60))" "$((SECONDS%60))"
