import os
import sys
import argparse
import yaml
import logging
from typing import Dict, List

import torch
import open_clip # Assuming RemoteCLIP uses open_clip

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from logger import loggers # Assuming logger.py is in root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Placeholder for LLM Interaction ---
# Replace this with your actual LLM API call or local model loading/inference
def get_description_from_llm(class_name: str, llm_model_name: str = "GPT-3.5-turbo") -> str:
    """
    Generates a high-quality description for a class name using an LLM.
    (This is a placeholder - implement your LLM logic here)
    """
    logger.info(f"Querying LLM ({llm_model_name}) for description of: {class_name}")

    # --- Start Placeholder ---
    # Example Prompt Engineering:
    prompt = (
        f"Provide a detailed and discriminative description suitable for radar target recognition "
        f"of a '{class_name}'. Focus on physical characteristics that might influence its radar "
        f"signature (e.g., size, shape, key components, materials if known), even if not directly "
        f"visible in a 1D HRRP. Aim for a concise paragraph."
        # Example Refinement: "Avoid mentioning specific radar frequencies unless relevant."
        # Example Refinement: "Emphasize differences from similar targets."
    )

    # Simulate LLM call (Replace with actual API call)
    # import openai
    # openai.api_key = "YOUR_API_KEY"
    # try:
    #     response = openai.ChatCompletion.create(
    #         model=llm_model_name,
    #         messages=[{"role": "user", "content": prompt}],
    #         max_tokens=150,
    #         temperature=0.5,
    #     )
    #     description = response.choices[0].message.content.strip()
    # except Exception as e:
    #     logger.error(f"LLM API call failed for {class_name}: {e}")
    #     description = f"A standard radar target known as {class_name}." # Fallback

    # Simple placeholder response:
    base_description = f"A radar target identified as {class_name}."
    if class_name == "F22":
        description = f"The {class_name} is a fifth-generation stealth fighter aircraft with diamond-shaped wings and canted twin tails, designed for low observability."
    elif class_name == "T72":
         description = f"The {class_name} is a main battle tank known for its low profile, rounded turret, and long main gun."
    elif class_name == "CAR":
         description = f"A standard civilian {class_name}, typically a four-wheeled passenger vehicle with a metallic body."
    else:
        description = base_description
    logger.info(f"Generated description: {description}")
    # --- End Placeholder ---

    return description

def load_config(config_path):
    """Loads YAML config file."""
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.error(f"Error loading YAML config: {exc}")
            sys.exit(1)
    return config

def main(config):
    """Generates and saves semantic features for target classes."""

    log_dir = os.path.join(config['paths']['logs'], 'semantics_generation')
    output_dir = os.path.dirname(config['semantics']['feature_path'])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    log = loggers(os.path.join(log_dir, f"generate_{config['semantics']['generation']['text_type']}"))
    log.info("Starting semantic feature generation...")
    log.info(f"Config: {config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Load Foundation Model (Text Encoder Part) ---
    fm_config = config['model']['foundation_model']
    log.info(f"Loading Foundation Model: {fm_config['name']} ({fm_config['variant']})")
    try:
        # Assuming RemoteCLIP uses open_clip interface
        if fm_config['name'] == 'RemoteCLIP':
            model, _, _ = open_clip.create_model_and_transforms(fm_config['variant'], pretrained='openai') # Or specify path if local
            tokenizer = open_clip.get_tokenizer(fm_config['variant'])
            text_encoder = model.to(device).eval() # Use the full model for encoding
            # Freeze model parameters
            for param in text_encoder.parameters():
                param.requires_grad = False
        else:
            log.error(f"Unsupported foundation model for text encoding: {fm_config['name']}")
            # Add logic here if using SARATR-X or other models that might need adaptation
            sys.exit(1)
        log.info("Foundation model text encoder loaded.")
    except Exception as e:
        log.error(f"Failed to load foundation model: {e}")
        sys.exit(1)

    # --- Get Class List ---
    all_classes = sorted(list(set(config['data']['base_classes'] + config['data']['novel_classes'])))
    log.info(f"Processing {len(all_classes)} classes: {all_classes}")

    # --- Generate Descriptions and Encode ---
    semantic_features: Dict[str, torch.Tensor] = {}
    llm_model_name = config['semantics']['generation'].get('llm', 'GPT-3.5-turbo') # Get LLM name from config

    for class_name in all_classes:
        # Step 1: Get description (using Semantic Evolution logic)
        if config['semantics']['generation']['text_type'] == 'name':
            description = f"A radar image of a {class_name}." # Simple template for 'name' type
        elif config['semantics']['generation']['text_type'] == 'definition':
            # Placeholder: Get definition from a dictionary or simple lookup
            definitions = {"T72": "Soviet main battle tank.", "F22": "American stealth fighter."}
            description = definitions.get(class_name, f"Definition of {class_name}.")
        elif config['semantics']['generation']['text_type'] in ['gpt', 'deepseek', 'claude']: # Assuming these use LLM
             description = get_description_from_llm(class_name, llm_model_name=llm_model_name)
        else:
             log.error(f"Unsupported text_type: {config['semantics']['generation']['text_type']}")
             description = class_name # Fallback to just the name

        # Step 2: Tokenize and Encode
        try:
            text_tokens = tokenizer([description]).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                text_feat = text_encoder.encode_text(text_tokens)
                # Normalize features (common practice for CLIP-like models)
                text_feat /= text_feat.norm(dim=-1, keepdim=True)
            semantic_features[class_name] = text_feat.squeeze(0).detach().cpu() # Store as (Dim,) tensor on CPU
            log.info(f"Encoded '{class_name}'. Feature shape: {semantic_features[class_name].shape}")

        except Exception as e:
            log.error(f"Error encoding description for {class_name}: {e}")
            # Optionally store a zero vector or skip
            semantic_features[class_name] = torch.zeros(fm_config['text_encoder_dim'])


    # --- Save Semantic Features ---
    output_path = config['semantics']['feature_path']
    try:
        torch.save({'semantic_feature': semantic_features}, output_path)
        log.info(f"Semantic features saved successfully to {output_path}")
    except Exception as e:
        log.error(f"Failed to save semantic features: {e}")

    log.info("Semantic feature generation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Semantic Features for HRRP Classes")
    parser.add_argument('--config', type=str, default='hrrp_fsl_config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found at {args.config}")
        sys.exit(1)

    configuration = load_config(args.config)
    main(configuration)
