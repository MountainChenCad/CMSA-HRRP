#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONFIG_FILE="configs/hrrp_fsl_config.yaml"
CONDA_ENV_NAME="CLIP"
# FSL settings for testing (can be overridden by test_fsl.py args if needed)
K_SHOTS_TO_TEST=(1 5) # Test both 1-shot and 5-shot

# --- Activate Environment ---
echo "Activating Conda environment: $CONDA_ENV_NAME..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME
echo "Environment activated."

# --- Verify Config ---
echo "--------------------------------------------------"
echo "Running CMSA-HRRP (Adapter Version) Pipeline"
echo "--------------------------------------------------"
echo "IMPORTANT: Ensure '$CONFIG_FILE' is configured with the desired"
echo "           'model.foundation_model.variant' and 'semantics.generation.text_type'"
echo "           for this run."
echo ""
# Extract current settings for confirmation (requires yq or similar, optional)
#VARIANT=$(yq e '.model.foundation_model.variant' $CONFIG_FILE)
#TEXT_TYPE=$(yq e '.semantics.generation.text_type' $CONFIG_FILE)
#echo "Current Config Settings: Variant=$VARIANT, TextType=$TEXT_TYPE"
read -p "Press Enter to continue after verifying the config file..." -r

# --- Stage 1: Train Adapter ---
echo "--- Stage 1: Training Adapter (train_adapter.py) ---"
python method/train_adapter.py --config $CONFIG_FILE
if [ $? -ne 0 ]; then echo "Adapter training failed."; conda deactivate; exit 1; fi
echo "Adapter training completed."
echo ""

# --- Stage 2: Compute Base Centers (z_V) ---
echo "--- Stage 2: Computing Base Centers (compute_centers.py) ---"
python scripts/compute_centers.py --config $CONFIG_FILE
if [ $? -ne 0 ]; then echo "Center computation failed."; conda deactivate; exit 1; fi
echo "Center computation completed."
echo ""

# --- Stage 3: Train SemAlign Module (h_F for z_V) ---
echo "--- Stage 3: Training SemAlign Module (train_semalign_stage3.py) ---"
python method/train_semalign_stage3.py --config $CONFIG_FILE
if [ $? -ne 0 ]; then echo "SemAlign module training failed."; conda deactivate; exit 1; fi
echo "SemAlign module training completed."
echo ""

# --- Stage 4: Few-Shot Evaluation ---
echo "--- Stage 4: Running Few-Shot Evaluation (test_fsl.py) ---"
# Runs the default kappa sweep defined in test_fsl.py (or config)
for k_shot in "${K_SHOTS_TO_TEST[@]}"; do
  echo "Running evaluation for ${k_shot}-shot..."
  python method/test_fsl.py --config $CONFIG_FILE --k_shot $k_shot
  if [ $? -ne 0 ]; then echo "FSL evaluation failed for ${k_shot}-shot."; conda deactivate; exit 1; fi
  echo "${k_shot}-shot evaluation completed."
  echo ""
done
echo "Few-shot evaluation finished."

# --- Deactivate Environment ---
echo "Deactivating Conda environment..."
conda deactivate
echo "CMSA-HRRP script finished successfully."

exit 0