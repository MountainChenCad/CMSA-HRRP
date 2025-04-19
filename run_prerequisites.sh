#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONFIG_FILE="configs/hrrp_fsl_config.yaml"
CONDA_ENV_NAME="CLIP"

# --- Activate Environment ---
echo "Activating Conda environment: $CONDA_ENV_NAME..."
source $(conda info --base)/etc/profile.d/conda.sh # Ensure conda command is available
conda activate $CONDA_ENV_NAME
echo "Environment activated."

# --- Generate Semantic Features ---
echo "--------------------------------------------------"
echo "Generating Semantic Features..."
echo "--------------------------------------------------"
echo "IMPORTANT: Ensure '$CONFIG_FILE' is configured with the desired"
echo "           'model.foundation_model.variant' (e.g., ViT-B-32, RN50, ViT-L-14)"
echo "           and 'semantics.generation.text_type' (e.g., detailed, llm_generated, name)"
echo "           BEFORE proceeding."
echo ""
read -p "Press Enter to continue after verifying the config file..." -r

echo "Running generate_semantics.py with config: $CONFIG_FILE"
python scripts/generate_semantics.py --config $CONFIG_FILE

if [ $? -eq 0 ]; then
  echo "Semantic feature generation completed successfully."
else
  echo "Semantic feature generation failed."
  conda deactivate
  exit 1
fi

# --- Deactivate Environment ---
echo "Deactivating Conda environment..."
conda deactivate
echo "Script finished."

exit 0