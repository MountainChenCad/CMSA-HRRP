import os
import sys
import argparse
import yaml
import logging # Import logging module
from typing import Dict, List

import torch
import open_clip # Assuming RemoteCLIP uses open_clip

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from logger import loggers # Assuming logger.py is in root

# --- Configure logging globally (can be done here or rely on logger.py setup) ---
# If logger.py configures the root logger, this basicConfig might not be strictly needed
# but it ensures logging works even if logger.py isn't called first elsewhere.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
# Get the logger for this module
logger = logging.getLogger(__name__) # Changed: Get logger instance directly

# --- Store the detailed descriptions ---
# Using the "Claude 3.7" descriptions provided previously
DETAILED_DESCRIPTIONS = {
    "EP-3E": "An ISAR radar image of a Lockheed EP-3E Aries II signals intelligence aircraft displaying its distinctive large fuselage and high wing configuration with four wing-mounted turboprop engines creating strong scattering points. The EP-3E's modified P-3 Orion airframe presents characteristic radar returns from its extended nose housing specialized signals collection equipment, prominent tail-mounted MAD boom, and multiple antenna arrays along the fuselage which create unique diffraction patterns and high-intensity radar reflections distinguishable from standard maritime patrol variants.",
    "F18": "An ISAR radar image of a Boeing F/A-18 Hornet revealing its mid-sized twin-engine tactical fighter signature with distinctive twin vertical stabilizers creating characteristic parallel returns. ISAR imaging captures the Hornet's unique wing geometry featuring leading-edge extensions (LEX) that generate strong edge diffraction, shoulder-mounted air intakes producing distinctive corner reflections, and canted twin tails that create identifiable multiple reflection points when illuminated from various aspect angles, making it readily distinguishable from single-tail fighters in radar returns.",
    "F22": "An ISAR radar image of a Lockheed Martin F-22 Raptor exhibiting its highly reduced radar cross-section due to advanced stealth features. The image shows minimal radar returns concentrated at carefully managed reflection points, with its distinctive planform alignment of leading and trailing edges designed to redirect radar energy away from the receiver. The F-22's angular chined fuselage, aligned edges, internal weapons carriage, and serrated panel joins create an extremely low-observable signature with characteristic diffuse scattering patterns that differentiate it from conventional fighters despite stealth countermeasures.",
    "F35": "An ISAR radar image of a Lockheed Martin F-35 Lightning II displaying its unique radar signature characterized by carefully managed radar returns with significantly reduced scattering centers compared to non-stealth aircraft. The F-35's distinctive feature in ISAR is its single-engine configuration with minimal specular reflections, optimized edge alignment, serpentine intake ducts, and radar absorbent materials that create a diffuse, low-intensity return pattern. Though larger than the F-22 in radar cross-section, it maintains stealth characteristics across multiple radar bands with recognizable faint returns from its lift fan housing (B variant) and diverterless supersonic inlet.",
    "IDF": "An ISAR radar image of an Indigenous Defense Fighter (AIDC F-CK-1 Ching-Kuo) displaying its compact multi-role fighter profile with distinctive twin-engine, twin-tail configuration. The ISAR return shows characteristic radar scattering from its unique hybrid design elements combining F-16 and F-18 influences, including the moderately swept wings with blended wing-root extensions, closely spaced twin vertical stabilizers, and side-mounted air intakes that create distinctive corner reflections and multiple scattering points when viewed from oblique angles, allowing differentiation from its American design influences.",
    "QQY": "An ISAR radar image of the RQ-4 Global Hawk high-altitude surveillance UAV showing its distinctive large wingspan (comparable to a 737) with high aspect ratio creating strong leading-edge returns. The ISAR signature reveals the Global Hawk's characteristic radar profile with minimal fuselage returns compared to its expansive wings, prominent V-tail configuration generating distinctive corner reflections, and bulbous nose housing advanced surveillance radar that creates a recognizable frontal radar cross-section pattern. Despite its size, the composite materials and smooth contours result in moderately reduced radar returns compared to conventional aircraft of similar dimensions.",
    "BSZ": "An ISAR radar image of the MQ-1 Predator UAV showing its distinctive slim fuselage and long-span, high-aspect-ratio wing configuration creating a characteristic narrow but elongated radar return. The ISAR signature captures the Predator's unique inverted V-tail that generates distinctive corner reflections, its centerline-mounted pusher propeller creating flash returns when blade positions align with radar pulses, and the ventral equipment pods housing sensor and weapons systems that produce localized high-intensity returns. The overall low radar cross-section is punctuated by these key scattering centers that enable positive identification.",
    "EA-18G": "An ISAR radar image of the Boeing EA-18G Growler electronic warfare aircraft displaying its F/A-18F Super Hornet airframe with distinctive modifications. The ISAR signature reveals characteristic radar returns from wing-tip mounted ALQ-99 jamming pods creating strong edge diffraction points, additional centerline electronic warfare pods producing concentrated returns, and the same twin vertical stabilizer configuration as the F-18 but with unique antenna arrays mounted on the tails and fuselage. These electronic warfare modifications create a distinctly different scattering pattern compared to standard F/A-18 variants when imaged with sufficient resolution.",
    "F2": "An ISAR radar image of a Mitsubishi F-2 multirole fighter showing its enlarged F-16 derived airframe with 25% larger wing area creating distinctive extended wing returns. The ISAR signature captures the F-2's characteristic radar profile with its significantly larger nose radome producing a more pronounced frontal return than the F-16, compound-curved bubble canopy creating unique specular reflections, and the distinctive longer fuselage with increased radar cross-section from both front and side aspects. The enlarged vertical stabilizer and co-cured composite wing structure also contribute to identifiable diffraction patterns different from its American predecessor.",
    "F15": "An ISAR radar image of a McDonnell Douglas F-15 Eagle displaying its distinctive large tactical fighter profile with twin engines and vertical stabilizers creating strong multiple reflection points. The ISAR signature shows characteristic returns from the F-15's large radar cross-section attributed to its broad wingspan with squared wingtips, widely spaced engine nacelles creating pronounced corner reflections, and substantial vertical twin tails generating parallel linear returns. The Eagle's conventional non-stealthy design with numerous surface discontinuities, right-angle joins, and exposed weapons hardpoints produces multiple high-intensity scattering centers that make it readily identifiable in radar imagery.",
    "F16": "An ISAR radar image of a General Dynamics F-16 Fighting Falcon exhibiting its compact single-engine fighter profile with distinctive blended wing-body configuration. The ISAR signature reveals characteristic returns from the F-16's unique ventral air intake creating a strong diffraction point, mid-mounted swept wings with moderately strong leading-edge returns, and single vertical stabilizer producing an asymmetric high-intensity return from oblique angles. The relatively small radar cross-section compared to twin-engine fighters is punctuated by the bubble canopy and nose radome creating recognizable reflection patterns that enable identification despite its relatively compact radar signature.",
    "HY2000": "An ISAR radar image of the Dassault Mirage 2000 displaying its distinct delta-wing configuration which creates a characteristic triangular radar return pattern with strong edge diffraction at the wing leading edges. The ISAR signature reveals the Mirage's unique absence of horizontal stabilizers, creating a clean rear aspect, along with its single vertical tail that generates asymmetric returns unlike twin-tail designs. The semi-recessed air intakes positioned on either side of the fuselage produce distinctive corner reflections, while the nose-mounted canard surfaces on certain variants create additional forward scattering points that help differentiate it from other delta-wing designs in radar imagery."
}

# Mapping from potential config names (including Chinese) to keys in DESCRIPTIONS
CLASS_NAME_MAP = {
    "捕食者": "BSZ",
    "幻影2000": "HY2000",
    "全球鹰": "QQY",
    "BSZ": "BSZ",
    "HY2000": "HY2000",
    "QQY": "QQY",
    "EP-3E": "EP-3E",
    "F18": "F18",
    "F22": "F22",
    "F35": "F35",
    "IDF": "IDF",
    "EA-18G": "EA-18G",
    "F2": "F2",
    "F15": "F15",
    "F16": "F16",
}

# --- Placeholder for LLM Interaction (Not used if using detailed descriptions) ---
def get_description_from_llm(class_name: str, llm_model_name: str = "GPT-3.5-turbo") -> str:
    """ Placeholder for LLM call - NOT actively used if detailed descriptions are sufficient. """
    global logger # Access the global logger
    logger.warning(f"LLM function called for {class_name}, but using placeholders/fallback.")
    # --- Start Placeholder ---
    prompt = f"Provide a detailed description for radar recognition of a '{class_name}'."
    # Simulate LLM call
    description = f"A radar target identified as {class_name}." # Basic fallback
    # --- End Placeholder ---
    return description

def get_description(class_name: str, text_type: str) -> str:
    """
    Retrieves or generates the description based on the text_type.
    Uses the detailed descriptions stored globally.
    """
    global logger # Access the global logger
    logger.info(f"Getting description for: {class_name} (Type: {text_type})")

    description_key = CLASS_NAME_MAP.get(class_name)

    if description_key is None:
         logger.warning(f"No mapping found for class name '{class_name}'. Using name as fallback.")
         description_key = class_name

    if text_type == 'name':
        description = f"An ISAR radar image of a {description_key}."
    elif text_type == 'definition':
        description = DETAILED_DESCRIPTIONS.get(description_key, f"Definition of {description_key}.")
        if description == f"Definition of {description_key}.":
             logger.warning(f"No detailed description found for '{description_key}'. Using basic definition.")
    elif text_type in ['gpt', 'deepseek', 'claude']: # Use the detailed description directly
        description = DETAILED_DESCRIPTIONS.get(description_key)
        if description is None:
            logger.error(f"Detailed description missing for key '{description_key}' derived from '{class_name}'. Falling back.")
            description = f"A radar target identified as {description_key}."
    else:
        logger.error(f"Unsupported text_type: {text_type}")
        description = description_key

    logger.debug(f"Using description: {description[:100]}...")
    return description

def load_config(config_path):
    """Loads YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            # Use global logger here too
            global logger
            logger.error(f"Error loading YAML config: {exc}")
            sys.exit(1)
    return config

def main(config):
    """Generates and saves semantic features for target classes."""

    # --- Setup Directories ---
    log_dir = os.path.join(config['paths']['logs'], 'semantics_generation')
    # Ensure output directory uses text_type from config for uniqueness
    output_filename = os.path.basename(config['semantics']['feature_path'])
    output_dir = os.path.dirname(config['semantics']['feature_path'])
    cfg_text_type = config['semantics']['generation']['text_type']
    if cfg_text_type not in output_filename:
         base, ext = os.path.splitext(output_filename)
         fm_variant_safe = config['model']['foundation_model']['variant'].replace('/', '-')
         new_filename = f"hrrp_semantics_{fm_variant_safe}_{cfg_text_type}{ext}"
         output_path = os.path.join(output_dir, new_filename)
         # Log this potential change AFTER initializing the logger
         # log.warning(f"Output path modified based on config text_type to: {output_path}")
    else:
         output_path = config['semantics']['feature_path']

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


    # --- Initialize Logger FIRST ---
    log = loggers(os.path.join(log_dir, f"generate_{cfg_text_type}")) # Use config text_type for log name
    log.info("Starting semantic feature generation...")
    log.info(f"Config: {config}")
    # Log the output path decision now that logger is initialized
    if output_path != config['semantics']['feature_path']:
         log.warning(f"Output path modified based on config text_type to: {output_path}")
    else:
         log.info(f"Output path set to: {output_path}")

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # --- Load Foundation Model (Text Encoder Part) ---
    fm_config = config['model']['foundation_model']
    log.info(f"Loading Foundation Model: {fm_config['name']} ({fm_config['variant']})")
    try:
        if fm_config['name'] == 'RemoteCLIP':
            # --- MODIFICATION START: Construct local weights path dynamically ---
            fm_variant = fm_config['variant']  # e.g., 'ViT-B-32'

            # Construct the expected filename based on your provided pattern
            weights_filename = f"RemoteCLIP-{fm_variant}.pt"

            # Construct the full path using the base checkpoint dir from config
            # Assumes weights are in '<checkpoints>/foundation_models/' subdirectory
            base_checkpoint_dir = config['paths'].get('checkpoints', './checkpoints')  # Default if not in config
            local_weights_path = os.path.join(
                base_checkpoint_dir,
                'foundation_models',  # Standard subdirectory for these weights
                weights_filename
            )

            # Check if the constructed path exists
            if not os.path.exists(local_weights_path):
                log.error(f"Dynamically constructed weights path does not exist: {local_weights_path}")
                log.error(
                    f"Please ensure the weights file '{weights_filename}' is placed in the directory: '{os.path.join(base_checkpoint_dir, 'foundation_models')}'")
                sys.exit(1)

            log.info(f"Loading weights from dynamically constructed local path: {local_weights_path}")
            pretrained_weights = local_weights_path  # Use the constructed local path
            # --- MODIFICATION END ---

            model, _, _ = open_clip.create_model_and_transforms(fm_config['variant'], pretrained=pretrained_weights)
            tokenizer = open_clip.get_tokenizer(fm_config['variant'])
            text_encoder = model.to(device).eval()  # Use the full model for encoding
            for param in text_encoder.parameters():
                param.requires_grad = False
        else:
            log.error(f"Unsupported foundation model for text encoding: {fm_config['name']}")
            sys.exit(1)
        log.info("Foundation model text encoder loaded.")
    except Exception as e:
        log.error(f"Failed to load foundation model: {e}")
        # Print traceback for more details during debugging
        import traceback
        log.error(traceback.format_exc())
        sys.exit(1)

    # --- Get Class List ---
    all_classes_from_config = sorted(list(set(config['data']['base_classes'] + config['data']['novel_classes'])))
    log.info(f"Processing {len(all_classes_from_config)} classes from config: {all_classes_from_config}")

    # --- Generate Descriptions and Encode ---
    semantic_features: Dict[str, torch.Tensor] = {}
    text_type = config['semantics']['generation']['text_type']

    for class_name in all_classes_from_config:
        description = get_description(class_name, text_type)
        try:
            text_tokens = tokenizer([description]).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                text_feat = text_encoder.encode_text(text_tokens)
                text_feat /= text_feat.norm(dim=-1, keepdim=True)
            semantic_features[class_name] = text_feat.squeeze(0).detach().cpu()
            log.info(f"Encoded '{class_name}'. Feature shape: {semantic_features[class_name].shape}")

        except Exception as e:
            log.error(f"Error encoding description for {class_name}: {e}")
            semantic_features[class_name] = torch.zeros(fm_config['text_encoder_dim'])


    # --- Save Semantic Features ---
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
        # Use standard logging BEFORE logger is fully configured if needed
        logging.error(f"Configuration file not found at {args.config}")
        sys.exit(1)

    configuration = load_config(args.config)
    main(configuration)
