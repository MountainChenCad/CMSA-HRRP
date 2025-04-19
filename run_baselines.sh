#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
CONFIG_FILE="configs/hrrp_fsl_config.yaml"
CONDA_ENV_NAME="CLIP"
K_SHOTS_TO_TEST=(1 5) # Test both 1-shot and 5-shot

# --- Activate Environment ---
echo "Activating Conda environment: $CONDA_ENV_NAME..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME
echo "Environment activated."

# --- Verify Config (for text_type needed for fusion) ---
echo "--------------------------------------------------"
echo "Running Baseline Pipelines"
echo "--------------------------------------------------"
echo "IMPORTANT: Ensure '$CONFIG_FILE' is configured with the desired"
echo "           'semantics.generation.text_type' for the 1DCNN+Semantics baseline."
echo "           (The baseline CNN training itself is independent of VLM/text_type)."
echo ""
read -p "Press Enter to continue after verifying the config file..." -r


# --- Baseline 1: ProtoNet (1D CNN) ---
echo "--- Baseline 1: ProtoNet (1D CNN) ---"

echo "Step 1.1: Training Baseline 1D CNN (train_baseline_cnn.py)..."
# Run only if checkpoint doesn't exist? Or just run to ensure it's up-to-date.
python method/train_baseline_cnn.py --config $CONFIG_FILE
if [ $? -ne 0 ]; then echo "Baseline CNN training failed."; conda deactivate; exit 1; fi
echo "Baseline CNN training completed."
echo ""

echo "Step 1.2: Evaluating ProtoNet (test_protonet_1dcnn.py)..."
for k_shot in "${K_SHOTS_TO_TEST[@]}"; do
  echo "Running ProtoNet evaluation for ${k_shot}-shot..."
  # Using default latest checkpoint from train_baseline_cnn.py
  python method/test_protonet_1dcnn.py --config $CONFIG_FILE --k_shot $k_shot --classifier cosine # Or euclidean
  if [ $? -ne 0 ]; then echo "ProtoNet evaluation failed for ${k_shot}-shot."; conda deactivate; exit 1; fi
  echo "ProtoNet ${k_shot}-shot evaluation completed."
  echo ""
done
echo "ProtoNet baseline evaluation finished."
echo ""


# --- Baseline 2: 1D CNN + Semantics ---
echo "--- Baseline 2: 1D CNN + Semantics ---"
# Assumes Baseline 1D CNN is already trained from above step

echo "Step 2.1: Computing 1D CNN Base Centers (z_H) (compute_centers_1dcnn.py)..."
# Uses the latest trained baseline CNN
python scripts/compute_centers_1dcnn.py --config $CONFIG_FILE
if [ $? -ne 0 ]; then echo "1D CNN Center computation failed."; conda deactivate; exit 1; fi
echo "1D CNN Center computation completed."
echo ""

echo "Step 2.2: Training Fusion Module for 1D CNN (train_fusion_1dcnn.py)..."
# Uses the latest trained baseline CNN, its centers, and the z_T corresponding to current config's text_type
python method/train_fusion_1dcnn.py --config $CONFIG_FILE
if [ $? -ne 0 ]; then echo "Baseline Fusion module training failed."; conda deactivate; exit 1; fi
echo "Baseline Fusion module training completed."
echo ""

echo "Step 2.3: Evaluating 1D CNN + Semantics (test_1dcnn_semantics.py)..."
# Uses latest baseline CNN and its fusion module
for k_shot in "${K_SHOTS_TO_TEST[@]}"; do
  echo "Running 1DCNN+Semantics evaluation for ${k_shot}-shot..."
  python method/test_1dcnn_semantics.py --config $CONFIG_FILE --k_shot $k_shot
  if [ $? -ne 0 ]; then echo "1DCNN+Semantics evaluation failed for ${k_shot}-shot."; conda deactivate; exit 1; fi
  echo "1DCNN+Semantics ${k_shot}-shot evaluation completed."
  echo ""
done
echo "1D CNN + Semantics baseline evaluation finished."


# --- Deactivate Environment ---
echo "Deactivating Conda environment..."
conda deactivate
echo "Baselines script finished successfully."

exit 0