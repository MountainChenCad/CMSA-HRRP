import os
import glob
from typing import List, Optional, Callable, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import logging

logger = logging.getLogger(__name__)

class HRRPDataset(Dataset):
    """
    Dataset class for loading HRRP data from .mat files.
    Handles simulated and measured data with different variable names and lengths.
    """
    def __init__(self,
                 root_dirs: List[str],
                 split: str, # 'base' or 'novel'
                 classes: List[str],
                 target_length: int,
                 normalization: str = 'log_max',
                 transform: Optional[Callable] = None,
                 phase_info: str = 'magnitude'): # 'magnitude', 'complex', 'phase_only'
        """
        Args:
            root_dirs (List[str]): List of directories containing HRRP .mat files.
            split (str): The dataset split ('base' or 'novel').
            classes (List[str]): List of target type names for this specific split.
            target_length (int): The desired length for all HRRP vectors (padding/truncation).
            normalization (str): Normalization method ('max', 'log_max', 'none').
            transform (Optional[Callable]): Optional transform to be applied on a sample.
            phase_info (str): How to handle complex data ('magnitude', 'complex', 'phase_only').
        """
        super().__init__()
        self.root_dirs = root_dirs
        self.split = split
        # Store class names exactly as provided in the config
        self.target_classes = classes
        self.target_length = target_length
        self.normalization = normalization.lower()
        self.transform = transform
        self.phase_info = phase_info.lower()

        self.filepaths: List[str] = []
        self.labels: List[int] = [] # Integer labels 0, 1, 2... specific to this split

        # Create mapping based on the order of classes provided for this split
        self.class_to_idx: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(self.target_classes)}
        self.idx_to_class: Dict[int, str] = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        self.targets = [] # Keep compatible with CategoriesSampler if needed

        self._scan_files()
        self.targets = self.labels # Assign labels to targets attribute

        logger.info(f"[{self.split} split] Found {len(self.filepaths)} samples for {len(self.target_classes)} classes: {self.target_classes}")
        logger.info(f"[{self.split} split] Class mapping (Split Index -> Name): {self.idx_to_class}")
        logger.info(f"[{self.split} split] Target length: {self.target_length}, Normalization: {self.normalization}, Phase Info: {self.phase_info}")


    def _scan_files(self):
        """ Scans root directories for .mat files matching the target classes. """
        found_classes = set()
        for root_dir in self.root_dirs:
            if not os.path.isdir(root_dir):
                logger.warning(f"Directory not found: {root_dir}. Skipping.")
                continue
            logger.info(f"Scanning directory: {root_dir}")
            # Use glob to find all .mat files recursively or directly
            # Adjust pattern if files are nested deeper
            for filepath in glob.glob(os.path.join(root_dir, '**/*.mat'), recursive=True):
                filename = os.path.basename(filepath)
                try:
                    # Extract target type based on naming convention 'TargetType_...'
                    # Important: Use the exact class name string from the config list
                    target_type = filename.split('_')[0]
                    if target_type in self.target_classes:
                        self.filepaths.append(filepath)
                        # Use the index assigned to this class *within this split's list*
                        self.labels.append(self.class_to_idx[target_type])
                        found_classes.add(target_type)
                except IndexError:
                    logger.warning(f"Could not parse target type from filename: {filename}. Skipping.")
                    continue
        # Verify all target classes were found
        missing_classes = set(self.target_classes) - found_classes
        if missing_classes:
            logger.warning(f"[{self.split} split] Did not find any files for classes: {missing_classes}")


    def _load_mat(self, filepath: str) -> Optional[np.ndarray]:
        """ Loads data from .mat file, handling different variable names. """
        try:
            mat_data = loadmat(filepath)
            if 'CoHH' in mat_data: # Simulated data
                hrrp = mat_data['CoHH'].flatten() # Ensure 1D array
                return hrrp
            elif 'data' in mat_data: # Measured data
                hrrp = mat_data['data'].flatten() # Ensure 1D array
                return hrrp
            else:
                # Try to find any key that might contain the data (less robust)
                possible_keys = [k for k in mat_data if not k.startswith('__')]
                if len(possible_keys) == 1:
                    logger.warning(f"Found potential data key '{possible_keys[0]}' in {filepath} (expected 'CoHH' or 'data').")
                    hrrp = mat_data[possible_keys[0]].flatten()
                    return hrrp
                else:
                    logger.warning(f"Could not find 'CoHH' or 'data' variable in {filepath}. Skipping. Keys found: {list(mat_data.keys())}")
                    return None
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}", exc_info=True)
            return None

    def _preprocess_hrrp(self, hrrp: np.ndarray) -> np.ndarray:
        """ Normalizes and adjusts length of the HRRP data. """

        # 1. Handle Complex Data
        if np.iscomplexobj(hrrp):
            if self.phase_info == 'magnitude':
                hrrp = np.abs(hrrp)
            elif self.phase_info == 'complex':
                # Keep complex, maybe separate real/imag later or use complex networks
                pass # Shape remains (L,) complex
            elif self.phase_info == 'phase_only':
                hrrp = np.angle(hrrp) # Shape (L,) real
            else:
                 logger.warning(f"Unknown phase_info '{self.phase_info}'. Defaulting to magnitude.")
                 hrrp = np.abs(hrrp)
        else:
             # Ensure it's float if it was real to begin with
             hrrp = hrrp.astype(np.float32)

        # Prepare for normalization: Use magnitude if data is complex, otherwise use data directly
        if self.phase_info == 'complex' and np.iscomplexobj(hrrp):
             hrrp_for_norm = np.abs(hrrp)
        else:
             hrrp_for_norm = hrrp # Should be real now if magnitude/phase_only or started real

        # 2. Normalization (applied to magnitude or real part)
        if self.normalization == 'max':
            max_val = np.max(hrrp_for_norm)
            if max_val > 1e-8:
                # Apply scaling to original data (could be complex or real)
                hrrp = hrrp / max_val
            else:
                hrrp = np.zeros_like(hrrp) # Avoid division by zero
        elif self.normalization == 'log_max':
             # Apply log to magnitude/real part, then min-max normalize
             # Important: This converts complex data to real log-magnitude
             hrrp_log = 10 * np.log10(np.maximum(hrrp_for_norm, 1e-10)) # Add epsilon for stability, ensure non-negative
             max_val = np.max(hrrp_log)
             min_val = np.min(hrrp_log)
             if max_val - min_val > 1e-8:
                 # Normalize the log-magnitude representation to [0, 1]
                 hrrp = (hrrp_log - min_val) / (max_val - min_val)
             else:
                 hrrp = np.zeros_like(hrrp_log) # Handle constant case

             if self.phase_info == 'complex':
                 logger.debug("Log normalization applied to magnitude. Phase info effectively lost.")
                 # Data is now real log-magnitude
                 # self.phase_info = 'magnitude' # Let's not change instance state here

        elif self.normalization != 'none':
            logger.warning(f"Unknown normalization method: {self.normalization}. Skipping normalization.")

        # Ensure hrrp is float32 after normalization if it became real
        if not np.iscomplexobj(hrrp):
            hrrp = hrrp.astype(np.float32)


        # 3. Adjust Length
        current_length = hrrp.shape[-1] # Get length from last dimension
        if current_length < self.target_length:
            # Pad
            pad_width = self.target_length - current_length
            pad_before = pad_width // 2
            pad_after = pad_width - pad_before
            # Pad the last dimension
            pad_spec = [(0, 0)] * (hrrp.ndim - 1) + [(pad_before, pad_after)]
            hrrp = np.pad(hrrp, pad_spec, 'constant', constant_values=0)

        elif current_length > self.target_length:
            # Truncate (center crop)
            start = (current_length - self.target_length) // 2
            end = start + self.target_length
            # Slice the last dimension
            slices = [slice(None)] * (hrrp.ndim - 1) + [slice(start, end)]
            hrrp = hrrp[tuple(slices)]


        # 4. Add channel dimension if needed (for typical CNN input)
        # If data is complex (and kept complex), it's still 1D. If real, add channel.
        if hrrp.ndim == 1:
             # Add channel dim -> (1, target_length)
             # Only do this if phase_info is not 'complex' which would imply we stacked C=2 earlier
             # Currently, complex data is kept as 1D complex array or converted to 1D real.
             hrrp = np.expand_dims(hrrp, axis=0)


        # Handle 'complex' case specifically if we decided to stack real/imag
        # Example (if stacking was desired):
        # if self.phase_info == 'complex' and np.iscomplexobj(hrrp) and hrrp.ndim == 1:
        #     hrrp_real = hrrp.real.astype(np.float32)
        #     hrrp_imag = hrrp.imag.astype(np.float32)
        #     hrrp = np.stack([hrrp_real, hrrp_imag], axis=0) # Shape becomes (2, target_length)
        # elif hrrp.ndim == 1: # If real, add channel dim
        #      hrrp = np.expand_dims(hrrp, axis=0)

        return hrrp

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        filepath = self.filepaths[index]
        label = self.labels[index] # This is the 0-based index for the split

        hrrp_raw = self._load_mat(filepath)

        if hrrp_raw is None:
            logger.error(f"Failed to load sample at index {index}, path {filepath}. Returning zero tensor.")
            # Return zero tensor of shape (1, target_length) - assuming magnitude/real output
            # Adjust channels if complex stacking is implemented
            num_channels = 1 # Default for magnitude/log_max/phase_only
            dummy_hrrp = torch.zeros((num_channels, self.target_length), dtype=torch.float32)
            # Return label 0 or handle appropriately
            return dummy_hrrp, 0

        hrrp_processed = self._preprocess_hrrp(hrrp_raw)

        # Ensure output is float32 tensor
        if np.iscomplexobj(hrrp_processed):
             # If complex processing is added later, handle complex tensor creation
             logger.warning("Complex data processing in __getitem__ not fully implemented for tensor conversion.")
             # For now, assume magnitude/real
             hrrp_tensor = torch.from_numpy(np.abs(hrrp_processed).astype(np.float32))
             if hrrp_tensor.ndim == 1: hrrp_tensor = hrrp_tensor.unsqueeze(0) # Add channel dim
        else:
            hrrp_tensor = torch.from_numpy(hrrp_processed.astype(np.float32))
            # Ensure channel dim exists if not added in preprocess
            if hrrp_tensor.ndim == 1: hrrp_tensor = hrrp_tensor.unsqueeze(0)


        if self.transform:
            hrrp_tensor = self.transform(hrrp_tensor)

        # Final check on shape
        if hrrp_tensor.shape[-1] != self.target_length or hrrp_tensor.ndim != 2:
            logger.warning(f"Unexpected tensor shape {hrrp_tensor.shape} for index {index}. Expected (C, {self.target_length}).")
            # Attempt to fix length? Padding/truncating again might be needed if preprocess failed.
            # Or just return the potentially wrong shape and let downstream handle/fail.

        return hrrp_tensor, label

# Example Usage (for testing):
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example config values (replace with your actual paths and classes from config)
    # Assuming config file exists at ../configs/hrrp_fsl_config.yaml
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), '../configs/hrrp_fsl_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        sim_path = config['data']['simulated_path']
        meas_path = config['data']['measured_path']
        base_cls = config['data']['base_classes']
        novel_cls = config['data']['novel_classes']
        target_len = config['data']['target_length']
        norm = config['data']['normalization']
    except Exception as e:
        print(f"Error loading config for example: {e}. Using dummy values.")
        sim_path = './datasets/simulated_hrrp'
        meas_path = './datasets/measured_hrrp'
        base_cls = ['F15', 'F16'] # Dummy
        novel_cls = ['F22', 'F18'] # Dummy
        target_len = 1000
        norm = 'log_max'


    # Create dummy files for testing if they don't exist
    os.makedirs(sim_path, exist_ok=True)
    os.makedirs(meas_path, exist_ok=True)
    # Check only one path to simplify dummy creation trigger
    if not glob.glob(os.path.join(sim_path, '*.mat')) and not glob.glob(os.path.join(meas_path, '*.mat')):
        print("Creating dummy .mat files for testing...")
        from scipy.io import savemat
        dummy_sim_data = np.random.rand(1000, 1) + 1j * np.random.rand(1000, 1)
        dummy_meas_data = np.random.rand(500, 1)
        # Use class names from config for dummy files
        for cls in base_cls + novel_cls:
             # Determine if simulated or measured based on path (simple heuristic)
             is_sim = 'simulated' in sim_path # Crude check, adjust if needed
             path = sim_path if is_sim else meas_path
             var_name = 'CoHH' if is_sim else 'data'
             data_to_save = dummy_sim_data if is_sim else dummy_meas_data
             # Adjust length of dummy data before saving
             data_len = data_to_save.shape[0]
             if data_len < target_len:
                  pad_w = target_len - data_len
                  data_to_save = np.pad(data_to_save, ((pad_w//2, pad_w - pad_w//2), (0,0)), 'constant')
             elif data_len > target_len:
                  start = (data_len - target_len) // 2
                  data_to_save = data_to_save[start:start+target_len, :]

             for i in range(5): # Create 5 dummy files per class
                 try:
                     savemat(os.path.join(path, f"{cls}_dummy_{i}_test.mat"), {var_name: data_to_save})
                 except Exception as save_e:
                      print(f"Warning: Could not save dummy file for {cls}: {save_e}")
        print("Dummy files created (or attempted).")


    print("--- Testing Base Split ---")
    try:
        base_dataset = HRRPDataset(root_dirs=[sim_path, meas_path], split='base', classes=base_cls,
                                   target_length=target_len, normalization=norm, phase_info='magnitude')
        if len(base_dataset) > 0:
            sample, label = base_dataset[0]
            print(f"Sample shape: {sample.shape}, Label: {label} ({base_dataset.idx_to_class[label]})")
            print(f"Min: {sample.min()}, Max: {sample.max()}")
            # Check another sample
            if len(base_dataset) > 1:
                 sample2, label2 = base_dataset[len(base_dataset)//2]
                 print(f"Middle sample shape: {sample2.shape}, Label: {label2} ({base_dataset.idx_to_class[label2]})")
        else:
            print(f"Base dataset is empty. Check path '{sim_path}', '{meas_path}' and classes {base_cls}.")
    except Exception as e:
        print(f"Error creating base dataset: {e}")

    print("\n--- Testing Novel Split ---")
    try:
        novel_dataset = HRRPDataset(root_dirs=[sim_path, meas_path], split='novel', classes=novel_cls,
                                    target_length=target_len, normalization=norm, phase_info='magnitude')
        if len(novel_dataset) > 0:
            sample, label = novel_dataset[0]
            print(f"Sample shape: {sample.shape}, Label: {label} ({novel_dataset.idx_to_class[label]})")
        else:
            print(f"Novel dataset is empty. Check path '{sim_path}', '{meas_path}' and classes {novel_cls}.")
    except Exception as e:
        print(f"Error creating novel dataset: {e}")