#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONFIG_FILE="configs/hrrp_fsl_config.yaml"
CONFIG_BACKUP="${CONFIG_FILE}.bak"
CONDA_ENV_NAME="CLIP"      # Replace with your actual Conda environment name if different
K_SHOTS_TO_TEST=(1 5 10 20)      # Test k-shot scenarios defined here within test_fsl.py if needed

# Ablation settings: Description and corresponding yq update command
# CORRECTED yq SYNTAX for Go version: yq -i -y 'expression' filename
# This syntax is required based on the error message.
declare -A ABLATION_SETTINGS
ABLATION_SETTINGS=(
    ["contrastive_true"]="yq -i -y '.training.adapter_loss.use_contrastive = true' $CONFIG_FILE"
#    ["contrastive_false"]="yq -i -y '.training.adapter_loss.use_contrastive = false' $CONFIG_FILE"
    # Add more ablation settings here if needed, e.g., for different text types:
    # ["text_type_name"]="yq -i -y '.semantics.generation.text_type = \"name\"' $CONFIG_FILE"
    # ["text_type_detailed"]="yq -i -y '.semantics.generation.text_type = \"detailed\"' $CONFIG_FILE"
)

# --- Prerequisites Check ---
# Check for yq
if ! command -v yq &> /dev/null; then
    echo "ERROR: 'yq' command not found. Please install yq."
    echo "This script likely expects the Go implementation (by Mike Farah)."
    echo "See: https://github.com/mikefarah/yq/"
    exit 1
fi
# Check for jq - less critical if using Go yq, but included for completeness
# if ! command -v jq &> /dev/null; then
#     echo "Warning: 'jq' command not found."
# fi
if ! command -v conda &> /dev/null; then
    echo "ERROR: 'conda' command not found. Is Conda installed and configured?"
    exit 1
fi


# --- Activate Environment ---
echo "Activating Conda environment: $CONDA_ENV_NAME..."
# Source conda.sh to make conda command available in script
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Failed to source conda.sh."; exit 1; }
conda activate "$CONDA_ENV_NAME" || { echo "Failed to activate Conda environment '$CONDA_ENV_NAME'."; exit 1; }
echo "Environment activated."
echo ""

# --- Backup Original Config ---
echo "Backing up original config file to $CONFIG_BACKUP..."
# Check if backup already exists; if so, maybe skip or handle differently?
if [ -f "$CONFIG_BACKUP" ]; then
    echo "Backup file $CONFIG_BACKUP already exists. Using existing backup."
else
    cp "$CONFIG_FILE" "$CONFIG_BACKUP"
fi


# --- Run Ablation Loop ---
for setting_name in "${!ABLATION_SETTINGS[@]}"; do
    echo ""
    echo "=================================================="
    echo "Running Ablation Setting: $setting_name"
    echo "=================================================="
    echo ""

    # Restore config from backup before applying modification
    echo "Restoring config from backup..."
    cp "$CONFIG_BACKUP" "$CONFIG_FILE"

    # Apply current ablation setting modification
    echo "Applying setting: ${ABLATION_SETTINGS[$setting_name]}"
    eval "${ABLATION_SETTINGS[$setting_name]}" # Execute the corrected yq command

    # --- Run Full Pipeline for this setting ---

    # 1. Generate Semantics (Good practice, especially if VLM/text type changes)
    echo "--- Stage 0: Generating Semantic Features (scripts/generate_semantics.py) ---"
    python scripts/generate_semantics.py --config "$CONFIG_FILE" || { echo "ERROR: Semantic generation failed for $setting_name."; exit 1; }
    echo "Semantic generation completed for $setting_name."
    echo ""

    # 2. Train Adapter
    echo "--- Stage 1: Training Adapter (method/train_adapter.py) ---"
    python method/train_adapter.py --config "$CONFIG_FILE" || { echo "ERROR: Adapter training failed for $setting_name."; exit 1; }
    echo "Adapter training completed for $setting_name."
    echo ""

    # 3. Compute Base Centers (z_V)
    echo "--- Stage 2: Computing Base Centers (scripts/compute_centers.py) ---"
    python scripts/compute_centers.py --config "$CONFIG_FILE" || { echo "ERROR: Center computation failed for $setting_name."; exit 1; }
    echo "Center computation completed for $setting_name."
    echo ""

    # 4. Train SemAlign Module
    echo "--- Stage 3: Training SemAlign Module (method/train_semalign_stage3.py) ---"
    python method/train_semalign_stage3.py --config "$CONFIG_FILE" || { echo "ERROR: SemAlign module training failed for $setting_name."; exit 1; }
    echo "SemAlign module training completed for $setting_name."
    echo ""

    # 5. Few-Shot Evaluation (Includes Visualization)
    echo "--- Stage 4: Running Few-Shot Evaluation & Visualization (method/test_fsl.py) ---"
    # Run evaluation for specified k-shots; test_fsl.py handles kappa sweep and visualizations
    EXIT_CODE_FSL=0
    for k_shot in "${K_SHOTS_TO_TEST[@]}"; do
      echo "  Running evaluation for ${k_shot}-shot..."
      # Run test_fsl.py, passing k_shot argument
      python method/test_fsl.py --config "$CONFIG_FILE" --k_shot "$k_shot"
      if [ $? -ne 0 ]; then
          echo "  ERROR: FSL evaluation failed for ${k_shot}-shot in setting $setting_name."
          EXIT_CODE_FSL=1
      fi
    done
    echo "FSL Evaluation finished for $setting_name."
    echo ""
    # Check if any FSL run failed for this setting
    if [ $EXIT_CODE_FSL -ne 0 ]; then
        echo "ERROR: One or more FSL evaluations failed for setting $setting_name."
        # Optionally exit immediately: exit 1
    fi

done

# --- Restore Original Config ---
echo "Restoring original config file from $CONFIG_BACKUP..."
# Use mv to overwrite the modified file with the backup
mv "$CONFIG_BACKUP" "$CONFIG_FILE"

# --- Deactivate Environment ---
echo "Deactivating Conda environment..."
conda deactivate
echo "Ablation script finished."

exit 0