# scripts/generate_semantics.py
import os
import sys
import argparse
import yaml
import logging
from typing import Dict, List, Any

import torch
import open_clip
import torch.nn.functional as F

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from logger import loggers
from utils import load_config, get_dynamic_paths # Import helpers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Placeholder for LLM Interaction ---
def get_description_from_llm(class_name_key: str, llm_model_name: str) -> str:
    """ Placeholder for actual LLM API call. """
    logger.warning(f"LLM function called for {class_name_key} with {llm_model_name}, but using placeholders.")
    # Replace with actual API call logic if needed
    return f"Placeholder LLM description for radar target {class_name_key} using {llm_model_name}."

def get_description(config_class_name: str, text_type: str, llm_name: str, descriptions_data: Dict[str, Any]) -> str:
    """
    Retrieves or generates the description based on the text_type and description data.
    """
    logger.info(f"Getting description for config class: {config_class_name} (Type: {text_type}, LLM: {llm_name})")

    class_mapping = descriptions_data.get('class_mapping', {})
    canonical_key = class_mapping.get(config_class_name)

    if canonical_key is None:
        logger.warning(f"No mapping found for config class name '{config_class_name}'. Using name as fallback key.")
        canonical_key = config_class_name # Fallback

    description = f"Radar target {canonical_key}." # Default fallback

    if text_type == 'name':
        description = f"An ISAR radar image of a {canonical_key}."
    elif text_type == 'detailed':
        desc_source = descriptions_data.get('manual_detailed', {})
        description = desc_source.get(canonical_key)
        if description is None:
            logger.warning(f"No 'manual_detailed' description found for key '{canonical_key}'. Using fallback.")
            description = f"Detailed description for radar target {canonical_key}."
    elif text_type == 'llm_generated':
        # Construct the key for the descriptions YAML based on the LLM name from config
        llm_key = llm_name.lower().replace('.', '') + '_generated'
        desc_source = descriptions_data.get(llm_key, {})
        description = desc_source.get(canonical_key)
        if description is None:
             # Fallback to placeholder LLM call if description not pre-generated in YAML
             logger.warning(f"No pre-generated description found for LLM '{llm_name}' and key '{canonical_key}' in YAML. Calling placeholder LLM function.")
             description = get_description_from_llm(canonical_key, llm_name)

    else:
        logger.error(f"Unsupported text_type: {text_type}. Using fallback.")

    logger.debug(f"Using description for '{canonical_key}': {description[:100]}...")
    return description

def load_descriptions(desc_file_path: str) -> Dict[str, Any]:
    """Loads the descriptions YAML file."""
    if not os.path.exists(desc_file_path):
         logger.error(f"Semantic description file not found at: {desc_file_path}")
         return {} # Return empty dict if file not found
    try:
        with open(desc_file_path, 'r', encoding='utf-8') as f:
            descriptions = yaml.safe_load(f)
        logger.info(f"Loaded semantic descriptions from: {desc_file_path}")
        return descriptions if descriptions else {}
    except yaml.YAMLError as exc:
        logger.error(f"Error loading descriptions YAML: {exc}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load or parse descriptions file {desc_file_path}: {e}")
        return {}


def main(config_path: str):
    """Generates and saves semantic features for target classes based on config."""
    config = load_config(config_path)
    dynamic_paths = get_dynamic_paths(config)

    # --- Setup Logging ---
    log_dir = os.path.dirname(dynamic_paths.get('adapter_log_dir', './logs/semantics_generation')) # Reuse adapter log structure
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = f"generate_semantics_{config['model']['foundation_model']['variant'].replace('/', '-')}_{config['semantics']['generation']['text_type']}"
    log = loggers(os.path.join(log_dir, log_file_name)) # Use specific log file
    log.info("Starting semantic feature generation...")
    log.info(f"Loaded configuration from: {config_path}")
    log.info(f"Using VLM variant: {config['model']['foundation_model']['variant']}")
    log.info(f"Using text type: {config['semantics']['generation']['text_type']}")
    log.info(f"Semantic features will be saved to: {dynamic_paths['semantic_features']}")

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Load Semantic Descriptions Data ---
    descriptions_data = load_descriptions(config['semantics']['description_file'])
    if not descriptions_data or 'class_mapping' not in descriptions_data:
         log.error("Failed to load descriptions or class_mapping from YAML. Exiting.")
         sys.exit(1)

    # --- Load Foundation Model (Text Encoder Part) ---
    fm_config = config['model']['foundation_model']
    vlm_weights_path = dynamic_paths['vlm_weights']
    log.info(f"Loading VLM: {fm_config['name']} ({fm_config['variant']})")
    try:
        if not os.path.exists(vlm_weights_path):
            log.error(f"VLM weights file not found: {vlm_weights_path}")
            log.error(f"Please ensure weights for variant '{fm_config['variant']}' exist at the expected location.")
            sys.exit(1)
        log.info(f"Loading weights from: {vlm_weights_path}")

        model, _, _ = open_clip.create_model_and_transforms(fm_config['variant'], pretrained=vlm_weights_path)
        tokenizer = open_clip.get_tokenizer(fm_config['variant'])
        text_encoder = model.to(device).eval()
        for param in text_encoder.parameters():
            param.requires_grad = False
        log.info("Foundation model text encoder loaded and frozen.")
    except Exception as e:
        log.error(f"Failed to load foundation model: {e}", exc_info=True)
        sys.exit(1)

    # --- Get Class List ---
    all_classes_from_config = sorted(list(set(config['data']['base_classes'] + config['data']['novel_classes'])))
    log.info(f"Processing {len(all_classes_from_config)} classes from config: {all_classes_from_config}")

    # --- Generate Descriptions and Encode ---
    semantic_features: Dict[str, torch.Tensor] = {}
    text_type = config['semantics']['generation']['text_type']
    llm_name = config['semantics']['generation'].get('llm', 'Manual') # Get LLM name for context

    for class_name in all_classes_from_config:
        description = get_description(class_name, text_type, llm_name, descriptions_data)
        try:
            text_tokens = tokenizer([description]).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                text_feat = text_encoder.encode_text(text_tokens)
                # Normalize features
                text_feat = F.normalize(text_feat) # Use normalize util
                # text_feat /= text_feat.norm(dim=-1, keepdim=True) # old way

            # Use the original config class name as the key in the output dict
            semantic_features[class_name] = text_feat.squeeze(0).detach().cpu()
            log.info(f"Encoded '{class_name}'. Feature shape: {semantic_features[class_name].shape}")

        except Exception as e:
            log.error(f"Error encoding description for {class_name}: {e}", exc_info=True)
            # Ensure feature dim matches config if creating zeros
            expected_dim = fm_config.get('text_encoder_dim', 512)
            semantic_features[class_name] = torch.zeros(expected_dim)


    # --- Save Semantic Features ---
    output_path = dynamic_paths['semantic_features']
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({'semantic_feature': semantic_features}, output_path)
        log.info(f"Semantic features saved successfully to {output_path}")
    except Exception as e:
        log.error(f"Failed to save semantic features: {e}", exc_info=True)

    log.info("Semantic feature generation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Semantic Features for HRRP Classes")
    parser.add_argument('--config', type=str, default='configs/hrrp_fsl_config.yaml', help='Path to the main configuration file')
    args = parser.parse_args()

    main(args.config)