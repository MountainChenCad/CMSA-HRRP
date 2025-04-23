#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONFIG_FILE="configs/hrrp_fsl_config.yaml"
CONDA_ENV_NAME="CLIP" # Replace with your actual Conda environment name if different

# --- Activate Environment ---
echo "Activating Conda environment: $CONDA_ENV_NAME..."
# Source conda.sh to make conda command available in script
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Failed to source conda.sh. Is Conda installed and configured?"; exit 1; }
conda activate "$CONDA_ENV_NAME" || { echo "Failed to activate Conda environment '$CONDA_ENV_NAME'."; exit 1; }
echo "Environment activated."

# --- Generate Semantic Features ---
echo "--------------------------------------------------"
echo "Running Prerequisite: Generate Semantic Features"
echo "--------------------------------------------------"
echo "This script will generate the semantic feature file based on settings in '$CONFIG_FILE'."
echo "IMPORTANT: Ensure '$CONFIG_FILE' is configured with the desired:"
echo "           - model.foundation_model.variant (e.g., ViT-B-32)"
echo "           - semantics.generation.text_type (e.g., llm_generated)"
echo "           - semantics.description_file points to the correct YAML"
echo "           BEFORE proceeding."
echo ""
# Optional: Add yq dependency check if uncommenting below
# if command -v yq &> /dev/null; then
#     VARIANT=$(yq e '.model.foundation_model.variant' $CONFIG_FILE)
#     TEXT_TYPE=$(yq e '.semantics.generation.text_type' $CONFIG_FILE)
#     echo "Current Config: Variant=$VARIANT, TextType=$TEXT_TYPE"
# else
#     echo "NOTE: 'yq' command not found. Cannot display current config settings automatically."
# fi
read -p "Press Enter to continue after verifying the config file..." -r

echo ""
echo "Running generate_semantics.py..."
python scripts/generate_semantics.py --config "$CONFIG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
  echo "Semantic feature generation completed successfully."
else
  echo "ERROR: Semantic feature generation failed."
  # Deactivate environment before exiting on error
  echo "Deactivating Conda environment..."
  conda deactivate
  exit 1
fi
echo ""

# --- Deactivate Environment ---
echo "Deactivating Conda environment..."
conda deactivate
echo "Prerequisite script finished."

exit 0