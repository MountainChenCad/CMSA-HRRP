# CMSA-HRRP: Cross-Modal Semantic Alignment for Few-Shot HRRP Recognition

This repository contains the code implementation for the CMSA-HRRP framework, designed for Few-Shot Learning (FSL) on High-Resolution Range Profile (HRRP) radar targets. The core idea is to leverage the semantic knowledge from pre-trained Vision-Language Models (VLMs), like RemoteCLIP, to enhance HRRP recognition when labeled data is scarce. This is achieved by aligning HRRP features into the VLM's semantic space and using semantically-enhanced prototypes for classification.

## Features

*   Implements the CMSA-HRRP framework for HRRP FSL.
*   Supports using pre-trained VLMs (e.g., RemoteCLIP via `open_clip`) as a source of semantic knowledge.
*   Includes "Semantic Evolution" script to generate high-quality text descriptions and encode them into features.
*   Provides separate training scripts for the alignment phase and evaluation scripts for FSL tasks.
*   Configurable via a central YAML file (`hrrp_fsl_config.yaml`).
*   Supports simulated and potentially real HRRP datasets in `.mat` format.

## Requirements

*   Python 3.8+
*   PyTorch (tested with 1.10+, CUDA recommended)
*   Other dependencies listed in `requirements.txt`.
*   (Optional) An API key for an LLM provider (e.g., OpenAI) if using GPT for Semantic Evolution, or a locally hosted LLM.
*   (Optional) `scikit-learn` for Logistic Regression baseline or clustering utilities.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://your-repo-url/CMSA-HRRP.git
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

4.  **Download VLM Weights (if necessary):**
    The `open_clip` library typically handles downloading pre-trained weights (like 'openai' weights for ViT-B/32). If you are using locally stored weights for RemoteCLIP or other models, ensure the path in `hrrp_fsl_config.yaml` or relevant scripts is correct.

## Dataset Preparation

1.  **HRRP Data:**
    *   Place your simulated HRRP `.mat` files in the directory specified by `data.simulated_path` in `hrrp_fsl_config.yaml` (default: `./datasets/simulated_hrrp/`).
    *   Place your measured HRRP `.mat` files in the directory specified by `data.measured_path` (default: `./datasets/measured_hrrp/`).
    *   **File Naming Convention:** Files must be named starting with the target type, followed by an underscore (`_`), e.g., `F22_scan001_angle30.mat`, `T72_run5_config2.mat`. The part before the first underscore is used as the class label.
    *   **`.mat` File Structure:**
        *   Simulated files should contain a variable named `CoHH` (expected shape `(L, 1)` or `(1, L)` or `(L,)`).
        *   Measured files should contain a variable named `data` (expected shape `(L, 1)` or `(1, L)` or `(L,)`).
        *   Lengths can differ; they will be padded/truncated to `data.target_length`.

2.  **Class Splits:**
    *   Edit `hrrp_fsl_config.yaml` to define which target types belong to `data.base_classes` (used for training alignment) and `data.novel_classes` (used for FSL evaluation). Ensure these names exactly match the prefixes in your `.mat` filenames.

## Configuration

Modify `hrrp_fsl_config.yaml` to set up your experiment:

*   **`data`**: Paths to datasets, class splits, target HRRP length, normalization method.
*   **`fsl`**: N-way, K-shot, query count, and number of test episodes.
*   **`model`**: Dimensions for the HRRP encoder, alignment/fusion modules, and specification of the foundation VLM (name, variant, text encoder dim). Ensure `text_encoder_dim` matches the chosen VLM variant.
*   **`training.alignment`**: Hyperparameters for training `f_H` and `h_A`.
*   **`semantics`**: Path where semantic features will be saved/loaded from, and parameters related to their generation (LLM model, text type).
*   **`paths`**: Base directories for saving checkpoints and logs.

## Usage

The typical workflow involves three main steps:

**Step 1: Generate Semantic Features**

This step uses the VLM's text encoder (and potentially an LLM) to create semantic feature vectors for all target classes defined in the config file.

```bash
python scripts/generate_semantics.py --config hrrp_fsl_config.yaml
```

*   This script will:
    *   Load class names from the config.
    *   (If `text_type` requires LLM) Call the LLM API (or local model) via the placeholder `get_description_from_llm` function in the script – **you need to implement this LLM call**.
    *   Load the specified VLM's text encoder (`model.foundation_model` section in config).
    *   Encode the descriptions (or names/definitions based on `semantics.generation.text_type`).
    *   Save the resulting features to the path specified in `semantics.feature_path`.
*   Make sure the `semantics.feature_path` in the config points to where you want the `.pth` file saved.

**Step 2: Train Alignment Module**

This step trains the HRRP Encoder (`f_H`), the Alignment Module (`h_A`), and the Fusion Module (`h_F`) using the base classes (`data.base_classes`). The goal is to align HRRP features with the pre-computed semantic features.

```bash
python method/train_alignment.py --config hrrp_fsl_config.yaml
```

*   This script will:
    *   Load the base HRRP dataset.
    *   Load the pre-computed semantic features (from Step 1).
    *   Initialize `f_H`, `h_A`, `h_F`.
    *   Train the models using the alignment loss (e.g., Cosine Embedding Loss or L1/MSE between `z'_H` and `z_T`) and reconstruction loss (L1 between `h_F` output and `z_T`).
    *   Save checkpoints periodically and the final model to the directory configured in `paths.checkpoints` (specifically under `alignment_module/`).

**Step 3: Evaluate Few-Shot Performance**

This step evaluates the trained models on FSL tasks constructed from the novel classes (`data.novel_classes`).

```bash
# Default run using kappa from config
python method/test_fsl.py --config hrrp_fsl_config.yaml

# Run without semantic enhancement (kappa=0)
python method/test_fsl.py --config hrrp_fsl_config.yaml --kappa 0

# Run the NoAlign baseline (bypasses h_A)
python method/test_fsl.py --config hrrp_fsl_config.yaml --no_align # kappa can also be set

# Test specific settings (e.g., 5-way 5-shot)
python method/test_fsl.py --config hrrp_fsl_config.yaml --n_way 5 --k_shot 5

# Use a specific checkpoint instead of 'latest.pth'
python method/test_fsl.py --config hrrp_fsl_config.yaml --checkpoint ./checkpoints/alignment_module/epoch_50.pth
```

*   This script will:
    *   Load the novel HRRP dataset.
    *   Load the pre-computed semantic features for novel classes.
    *   Load the trained weights for `f_H`, `h_A`, `h_F` from the specified checkpoint (defaults to `latest.pth` from Step 2).
    *   Sample FSL episodes according to the config or command-line arguments.
    *   Calculate prototypes (using specified `kappa` for fusion).
    *   Classify query samples and report the average accuracy and 95% confidence interval.
    *   Logs are saved under the directory configured in `paths.logs` (specifically under `fsl_testing/<setting_name>/`).

## Project Structure

```
├── configs/
│   └── hrrp_fsl_config.yaml  # Main configuration file
├── data/
│   ├── hrrp_dataset.py       # HRRP Dataset loader
│   └── samplers.py           # FSL episode sampler
├── datasets/                 # Default location for data (create these)
│   ├── simulated_hrrp/       # Place simulated .mat files here
│   └── measured_hrrp/        # Place measured .mat files here
├── logs/                     # Stores logs from training and testing
├── method/
│   ├── alignment.py          # Alignment and Fusion module definitions
│   ├── train_alignment.py    # Script for training alignment
│   ├── train_fsl.py          # Placeholder for separate meta-training (optional)
│   └── test_fsl.py           # Script for FSL evaluation
├── model/
│   └── hrrp_encoder.py       # HRRP Encoder definition (e.g., 1D CNN)
├── semantic_features/        # Default location for saved semantic features
│   └── hrrp_semantics_....pth
├── checkpoints/              # Stores trained model weights
│   └── alignment_module/
├── scripts/
│   └── generate_semantics.py # Script to create semantic features
├── utils.py                  # Utility functions (metrics, seeding, etc.)
├── logger.py                 # Logging setup
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Citation

If you find this code useful for your research, please consider citing:

```bibtex
@misc{cmsa_hrrp_code,
  author       = {Your Name/Group},
  title        = {CMSA-HRRP: Cross-Modal Semantic Alignment for Few-Shot HRRP Recognition - Code Implementation},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://repo-url/CMSA-HRRP}}
}
```

## License

(Specify your license here, e.g., MIT, Apache 2.0)
