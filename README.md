# CMSA-HRRP (Adapter Version): Cross-Modal Semantic Adaption for Few-Shot HRRP Recognition

This repository contains the code implementation for CMSA-HRRP, a framework designed for Few-Shot Learning (FSL) on High-Resolution Range Profile (HRRP) radar targets using Vision-Language Models (VLMs). This version implements the approach where 1D HRRP signals are adapted into 2D pseudo-images via a trainable adapter, processed by a frozen VLM visual encoder (e.g., RemoteCLIP's ViT), and aligned with semantic features from the VLM's text encoder. **This version includes an optional visual-specific contrastive loss during adapter training to enhance feature discriminability.**

## Features

*   Implements an HRRP FSL framework using a 1D-to-2D Adapter + VLM Visual Encoder.
*   Includes a trainable `HRPPtoPseudoImage` adapter module (MLP+Reshape).
*   Adapter training uses a combination of cross-modal alignment loss (visual vs. text features) and an optional visual-specific contrastive loss (visual vs. visual features).
*   Leverages frozen pre-trained VLM visual and text encoders (e.g., RemoteCLIP via `open_clip`).
*   Includes "Semantic Evolution" script (`generate_semantics.py`) using pre-defined descriptions (from `configs/semantic_descriptions.yaml`) and encodes them into target semantic features $z_T$.
*   Provides separate training scripts for the adapter (`train_adapter.py`), baseline CNN (`train_baseline_cnn.py`), SemAlign fusion module for adapter (`train_semalign_stage3.py`), SemAlign fusion module for baseline (`train_fusion_1dcnn.py`), and scripts for computing class centers (`compute_centers.py`, `compute_centers_1dcnn.py`).
*   Includes evaluation scripts for the main method (`test_fsl.py`), baseline ProtoNet (`test_protonet_1dcnn.py`), and baseline 1D CNN + Semantics (`test_1dcnn_semantics.py`).
*   Configurable via a central YAML file (`configs/hrrp_fsl_config.yaml`).
*   Supports simulated and potentially real HRRP datasets in `.mat` format.
*   Supports semantic fusion ($\kappa > 0$) during FSL testing for prototype enhancement.
*   Dynamic path generation for checkpoints, logs, and features based on config (`utils.py`).

## Requirements

*   Python 3.8+
*   PyTorch (tested with 1.10+, CUDA recommended)
*   `PyYAML`, `numpy`, `scipy`, `tqdm`, `open_clip_torch`
*   (Optional) `scikit-learn` for Logistic Regression baseline in `utils.py` (not primary) and FSL baselines if implemented.
*   (Optional) `tensorboard` for logging visualization.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url> CMSA-HRRP
    cd CMSA-HRRP
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt # Create this file based on imports
    # Or: pip install torch torchvision torchaudio open_clip_torch PyYAML numpy scipy tqdm scikit-learn tensorboard
    ```
4.  **Download VLM Weights:**
    *   Download the pre-trained weights (`.pt` file) for the desired RemoteCLIP variant (e.g., `RemoteCLIP-ViT-B-32.pt`).
    *   Place the downloaded file in `./checkpoints/foundation_models/`. The scripts will find it based on the `model.foundation_model.variant` in the config.

## Dataset Preparation

1.  **HRRP Data:**
    *   Place simulated HRRP `.mat` files in `./datasets/simulated_hrrp/`.
    *   Place measured HRRP `.mat` files in `./datasets/measured_hrrp/`.
    *   **File Naming:** `TargetType_*.mat` (e.g., `F22_scan001.mat`). Ensure `TargetType` exactly matches the names listed in `data.base_classes` and `data.novel_classes` in `hrrp_fsl_config.yaml`.
    *   **`.mat` Structure:** Simulated: variable `CoHH`; Measured: variable `data`. Lengths adjusted to `data.target_length`.
2.  **Class Splits:** Edit `configs/hrrp_fsl_config.yaml` to define `data.base_classes` and `data.novel_classes`.
3.  **Semantic Descriptions:** Edit `configs/semantic_descriptions.yaml` to provide:
    *   `class_mapping`: Maps class names from `hrrp_fsl_config.yaml` to keys used in description sections.
    *   Description sections (e.g., `manual_detailed`, `<llm_name>_generated`) containing descriptions for the keys defined in `class_mapping`.

## Configuration (`configs/hrrp_fsl_config.yaml`)

Modify the YAML file:

*   **`data`**: Paths (`datasets_base`), class splits, `target_length`, `normalization`.
*   **`fsl`**: N-way, K-shot, `q_query`, `test_episodes`, `classifier_temperature`, `test_kappa_values` (optional sweep).
*   **`model`**:
    *   `adapter_1d_to_2d`: `intermediate_dim`, `activation`.
    *   `fusion_module`: `hidden_dim`, `kappa` (default for testing).
    *   `foundation_model`: `name` ('RemoteCLIP'), `variant` (e.g., 'ViT-B-32'), `visual_encoder_dim`, `text_encoder_dim`. **Ensure dims match the chosen VLM variant!**
    *   `baseline_cnn`: (Optional) Define params for the baseline `HRRPEncoder`.
*   **`training`**: `epochs`, `batch_size`, `lr`, `optimizer`.
    *   `adapter_loss`: `alignment_loss_type`, `use_contrastive`, `lambda_v2v`, `temperature_v`.
    *   `semalign_loss`: `loss_type`, `dropout_semalign`.
    *   `baseline_cnn_loss`: `loss_type`.
*   **`semantics`**: `description_file`, `generation` (`llm`, `text_type`).
*   **`paths`**: Base directories for `checkpoints`, `logs`, `semantic_features_dir`, `datasets_base`. Specific paths are derived dynamically.
*   **`num_workers`**, **`seed`**.
*   **`baseline_experiment_name`**: (Optional) Identifier for baseline model checkpoints/logs.

## How to Use (Step-by-Step)

**Prerequisites:** Ensure data is prepared, VLM weights are downloaded, and `configs/hrrp_fsl_config.yaml` and `configs/semantic_descriptions.yaml` are correctly configured.

**Step 1: Generate Semantic Features ($z_T$)**
Encodes text descriptions (specified by `semantics.generation.text_type` in config) using the frozen VLM text encoder.
```bash
python scripts/generate_semantics.py --config configs/hrrp_fsl_config.yaml