# CMSA-HRRP (Adapter Version): Cross-Modal Semantic Alignment for Few-Shot HRRP Recognition via Pseudo-Images

This repository contains the code implementation for a version of the CMSA-HRRP framework designed for Few-Shot Learning (FSL) on High-Resolution Range Profile (HRRP) radar targets. **This specific version implements the approach where 1D HRRP signals are adapted into 2D pseudo-images, which are then processed by a pre-trained Vision-Language Model's (VLM) visual encoder (e.g., RemoteCLIP's ViT).** The goal is to align these resulting visual features with the VLM's text/semantic embeddings to leverage its learned cross-modal knowledge for enhanced HRRP recognition when labeled data is scarce.

## Features

*   Implements an HRRP FSL framework using a 1D-to-2D Adapter + VLM Visual Encoder.
*   Includes a trainable `HRPPtoPseudoImage` adapter module (using MLP+Reshape by default).
*   Leverages frozen pre-trained VLM visual and text encoders (e.g., RemoteCLIP via `open_clip`).
*   Includes "Semantic Evolution" script to generate high-quality text descriptions and encode them into target semantic features.
*   Provides separate training scripts for the adapter (`train_adapter.py`) and evaluation scripts for FSL tasks (`test_fsl.py`).
*   Configurable via a central YAML file (`configs/hrrp_fsl_config.yaml`).
*   Supports simulated and potentially real HRRP datasets in `.mat` format.
*   (Optional) Supports semantic fusion of visual features and text features for prototype enhancement.

## Requirements

*   Python 3.8+
*   PyTorch (tested with 1.10+, CUDA recommended)
*   Other dependencies listed in `requirements.txt`.
*   (Optional) An API key for an LLM provider (e.g., OpenAI) if using GPT for Semantic Evolution, or a locally hosted LLM.
*   (Optional) `scikit-learn` for Logistic Regression baseline.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://your-repo-url/CMSA-HRRP.git # Replace with your repo URL
    cd CMSA-HRRP
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download VLM Weights:**
    *   Download the pre-trained weights (`.pt` file) for the desired RemoteCLIP variant (e.g., `RemoteCLIP-ViT-B-32.pt`).
    *   Place the downloaded file in the directory specified by `paths.checkpoints` + `/foundation_models/` in the config file (default: `./checkpoints/foundation_models/`). The script `scripts/generate_semantics.py` and `method/train_adapter.py` will dynamically construct the path based on the `model.foundation_model.variant` in the config.

## Dataset Preparation

1.  **HRRP Data:**
    *   Place simulated HRRP `.mat` files in `./datasets/simulated_hrrp/` (or path from config).
    *   Place measured HRRP `.mat` files in `./datasets/measured_hrrp/` (or path from config).
    *   **File Naming:** `TargetType_*.mat` (e.g., `F22_scan001.mat`).
    *   **`.mat` Structure:** Simulated: variable `CoHH`; Measured: variable `data`. Lengths will be adjusted to `data.target_length`.

2.  **Class Splits:**
    *   Edit `configs/hrrp_fsl_config.yaml` to define `data.base_classes` and `data.novel_classes` using exact filename prefixes.

## Configuration

Modify `configs/hrrp_fsl_config.yaml` (ensure path is correct):

*   **`data`**: Paths, class splits, `target_length`, `normalization`.
*   **`fsl`**: N-way, K-shot, `q_query`, `test_episodes`.
*   **`model`**:
    *   `hrrp_encoder`: **(No longer directly used in main pipeline, but config structure might remain)**.
    *   `adapter_1d_to_2d` **(Add this section if customizing adapter):** Define parameters for `HRPPtoPseudoImage` if needed (e.g., `intermediate_channels`, `activation`). Defaults are in the code.
    *   `alignment_module`: **(No longer directly used)**.
    *   `fusion_module`: `hidden_dim`, `kappa` (set `kappa > 0` to enable fusion).
    *   `foundation_model`: `name` ('RemoteCLIP'), `variant` ('ViT-B-32', 'ViT-L-14', 'RN50'), `text_encoder_dim` (ensure this matches the chosen variant's CLIP output dimension).
*   **`training.alignment`**: **(Now used for Adapter Training)** Hyperparameters like `epochs`, `batch_size`, `lr`, `optimizer`, `loss_type` ('cosine' recommended for CLIP alignment).
*   **`semantics`**: `feature_path` (output/input), `generation` settings.
*   **`paths`**: Base directories for `checkpoints` and `logs`. Note that adapter checkpoints are now saved under `<checkpoints>/hrrp_adapter/`.

## Usage

**Step 1: Generate Semantic Features**

Create semantic feature vectors using the VLM's text encoder.

```bash
python scripts/generate_semantics.py --config configs/hrrp_fsl_config.yaml
```
*   Loads class names, generates/retrieves descriptions based on `semantics.generation.text_type`.
*   **Requires implemented LLM call in `get_description_from_llm` if using 'gpt' etc.**
*   Loads VLM text encoder (weights path constructed dynamically).
*   Encodes text and saves features to `semantics.feature_path`.

**Step 2: Train HRRP 1D-to-2D Adapter**

Train the `HRPPtoPseudoImage` adapter module using the base classes. The goal is to make the VLM visual encoder's output for the pseudo-image match the VLM text encoder's output for the class description.

```bash
# Rename train_alignment.py to train_adapter.py if you prefer
python method/train_alignment.py --config configs/hrrp_fsl_config.yaml # Or train_adapter.py
```
*   Loads base HRRP data and semantic features.
*   Loads **frozen** VLM visual (`f_V`) and text (`f_T`) encoders.
*   Initializes the `HRPPtoPseudoImage` adapter (`h_1D_to_2D`).
*   Trains **only the adapter** by minimizing the cosine distance between `normalize(f_V(h_1D_to_2D(x_H)))` and `normalize(z_T)`.
*   Saves adapter checkpoints to `<checkpoints>/hrrp_adapter/`.

**Step 3: Evaluate Few-Shot Performance**

Evaluate the trained adapter and frozen VLM visual encoder on FSL tasks using novel classes.

```bash
# Default run using kappa from config
python method/test_fsl.py --config configs/hrrp_fsl_config.yaml

# Run without semantic fusion (kappa=0)
python method/test_fsl.py --config configs/hrrp_fsl_config.yaml --kappa 0

# Test specific settings (e.g., 5-way 5-shot)
python method/test_fsl.py --config configs/hrrp_fsl_config.yaml --n_way 5 --k_shot 5

# Use a specific adapter checkpoint
python method/test_fsl.py --config configs/hrrp_fsl_config.yaml --checkpoint ./checkpoints/hrrp_adapter/best.pth
```
*   Loads novel HRRP data and semantic features.
*   Loads the **trained `HRPPtoPseudoImage` adapter** weights.
*   Loads the **frozen VLM visual encoder (`f_V`)**.
*   (Optional) Loads the `FusionModule` (`h_F`) weights if `kappa > 0`.
*   Samples FSL episodes.
*   For each episode:
    *   Generates pseudo-images using the adapter: `x_pseudo = h_1D_to_2D(x_H)`.
    *   Extracts visual features using the VLM: `z_V = normalize(f_V(x_pseudo))`.
    *   Calculates prototypes based on `z_V` (optionally fused with `z_T` using `h_F` if `kappa > 0`).
    *   Classifies query `z_V` features.
*   Reports average accuracy and confidence interval. Logs saved to `<logs>/fsl_testing_adapter/<setting_name>/`.

## Project Structure (Updated)

```
├── configs/
│   └── hrrp_fsl_config.yaml  # Main configuration file
├── data/
│   ├── hrrp_dataset.py       # HRRP Dataset loader
│   └── samplers.py           # FSL episode sampler
├── datasets/                 # Default location for data
│   ├── simulated_hrrp/
│   └── measured_hrrp/
├── logs/
│   ├── adapter_training/     # Logs for adapter training
│   └── fsl_testing_adapter/  # Logs for FSL evaluation (adapter approach)
├── method/
│   ├── alignment.py          # Contains FusionModule (h_F) definition
│   ├── train_alignment.py    # Script for training the adapter (consider renaming)
│   ├── train_fsl.py          # Placeholder (optional)
│   └── test_fsl.py           # Script for FSL evaluation (adapter approach)
├── model/
│   └── hrrp_adapter_1d_to_2d.py # Defines HRPPtoPseudoImage adapter (h_1D_to_2D)
├── semantic_features/        # Default location for saved semantic features
├── checkpoints/
│   ├── hrrp_adapter/         # Stores trained adapter weights (e.g., latest.pth, best.pth)
│   └── foundation_models/    # Place downloaded VLM .pt files here
├── scripts/
│   └── generate_semantics.py # Script to create semantic features
├── utils.py                  # Utility functions
├── logger.py                 # Logging setup
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Citation

(Keep citation section as before)

## License
