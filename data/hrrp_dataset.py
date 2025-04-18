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
        self.target_classes = classes
        self.target_length = target_length
        self.normalization = normalization.lower()
        self.transform = transform
        self.phase_info = phase_info.lower()

        self.filepaths: List[str] = []
        self.labels: List[int] = []
        self.class_to_idx: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(sorted(self.target_classes))}
        self.idx_to_class: Dict[int, str] = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        self.targets = [] # Keep compatible with CategoriesSampler if needed

        self._scan_files()
        self.targets = self.labels # Assign labels to targets attribute

        logger.info(f"[{self.split} split] Found {len(self.filepaths)} samples for {len(self.target_classes)} classes.")
        logger.info(f"[{self.split} split] Target length: {self.target_length}, Normalization: {self.normalization}, Phase Info: {self.phase_info}")


    def _scan_files(self):
        """ Scans root directories for .mat files matching the target classes. """
        for root_dir in self.root_dirs:
            if not os.path.isdir(root_dir):
                logger.warning(f"Directory not found: {root_dir}. Skipping.")
                continue
            logger.info(f"Scanning directory: {root_dir}")
            # Use glob to find all .mat files recursively or directly
            # Adjust pattern if files are nested deeper
            for filepath in glob.glob(os.path.join(root_dir, '*.mat')):
                filename = os.path.basename(filepath)
                try:
                    # Extract target type based on naming convention 'TargetType_...'
                    target_type = filename.split('_')[0]
                    if target_type in self.target_classes:
                        self.filepaths.append(filepath)
                        self.labels.append(self.class_to_idx[target_type])
                except IndexError:
                    logger.warning(f"Could not parse target type from filename: {filename}. Skipping.")
                    continue

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
                logger.warning(f"Could not find 'CoHH' or 'data' variable in {filepath}. Skipping.")
                return None
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None

    def _preprocess_hrrp(self, hrrp: np.ndarray) -> np.ndarray:
        """ Normalizes and adjusts length of the HRRP data. """

        # 1. Handle Complex Data
        if np.iscomplexobj(hrrp):
            if self.phase_info == 'magnitude':
                hrrp = np.abs(hrrp)
            elif self.phase_info == 'complex':
                # Keep complex, maybe separate real/imag later or use complex networks
                pass
            elif self.phase_info == 'phase_only':
                hrrp = np.angle(hrrp)
            else:
                 logger.warning(f"Unknown phase_info '{self.phase_info}'. Defaulting to magnitude.")
                 hrrp = np.abs(hrrp)
        else:
             # Ensure it's float if it was real to begin with
             hrrp = hrrp.astype(np.float32)

        # If complex, we might need to handle real/imag parts separately for standard networks
        # For simplicity now, assuming magnitude or real input for normalization
        if self.phase_info == 'complex':
             # Example: Stack real and imag as channels? Or just use magnitude for norm?
             # Stacking: hrrp = np.stack([hrrp.real, hrrp.imag], axis=0) # Becomes (2, L)
             # For now, let's use magnitude for normalization calculations if complex
             hrrp_for_norm = np.abs(hrrp) if np.iscomplexobj(hrrp) else hrrp
        else:
             hrrp_for_norm = hrrp


        # 2. Normalization (applied to magnitude or real part)
        if self.normalization == 'max':
            max_val = np.max(hrrp_for_norm)
            if max_val > 1e-8:
                hrrp = hrrp / max_val # Apply scaling to original (potentially complex) data
        elif self.normalization == 'log_max':
             # Apply log to magnitude/real part, then max normalize
             hrrp_log = 10 * np.log10(hrrp_for_norm + 1e-10) # Add epsilon for stability
             max_val = np.max(hrrp_log)
             min_val = np.min(hrrp_log)
             if max_val - min_val > 1e-8:
                 # Normalize the log-magnitude representation
                 hrrp = (hrrp_log - min_val) / (max_val - min_val) # Result is now real, normalized log-magnitude
             else:
                 hrrp = np.zeros_like(hrrp_log) # Handle constant case
             # Ensure phase info is handled correctly after log-magnitude processing
             if self.phase_info == 'complex':
                 logger.warning("Log normalization applied to magnitude. Phase info lost.")
                 self.phase_info = 'magnitude' # Update effective phase info
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
            # Pad symmetrically if possible, otherwise pad at the end
            pad_before = pad_width // 2
            pad_after = pad_width - pad_before
            # Handle potential multi-channel (e.g., complex as 2 channels)
            if hrrp.ndim == 1:
                 hrrp = np.pad(hrrp, (pad_before, pad_after), 'constant', constant_values=0)
            elif hrrp.ndim == 2: # e.g., (2, L) for complex
                 hrrp = np.pad(hrrp, ((0,0), (pad_before, pad_after)), 'constant', constant_values=0)

        elif current_length > self.target_length:
            # Truncate (center crop)
            start = (current_length - self.target_length) // 2
            end = start + self.target_length
            if hrrp.ndim == 1:
                hrrp = hrrp[start:end]
            elif hrrp.ndim == 2:
                hrrp = hrrp[:, start:end]


        # 4. Add channel dimension if output is 1D and phase_info is not complex (stacked)
        if hrrp.ndim == 1:
            hrrp = np.expand_dims(hrrp, axis=0) # Shape becomes (1, target_length)


        return hrrp

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        filepath = self.filepaths[index]
        label = self.labels[index]

        hrrp_raw = self._load_mat(filepath)

        if hrrp_raw is None:
            # Handle loading error, e.g., return a dummy sample or raise error
            logger.error(f"Failed to load sample at index {index}, path {filepath}. Returning zero tensor.")
            # Return zero tensor of shape (1, target_length) or (2, target_length) if complex expected
            num_channels = 2 if self.phase_info == 'complex' else 1
            dummy_hrrp = torch.zeros((num_channels, self.target_length), dtype=torch.float32)
            # Assign a valid label (e.g., 0) or handle appropriately downstream
            return dummy_hrrp, 0 # Or raise an exception

        hrrp_processed = self._preprocess_hrrp(hrrp_raw)
        hrrp_tensor = torch.from_numpy(hrrp_processed)

        if self.transform:
            # Note: Standard torchvision transforms might not work well on 1D signal data.
            # Custom transforms might be needed.
            hrrp_tensor = self.transform(hrrp_tensor)

        return hrrp_tensor, label

# Example Usage (for testing):
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example config values (replace with your actual paths and classes)
    sim_path = './datasets/simulated_hrrp'
    meas_path = './datasets/measured_hrrp'
    base_cls = ['BTR', 'T72'] # Example
    novel_cls = ['F22', 'CAR'] # Example
    target_len = 1000
    norm = 'log_max'

    # Create dummy files for testing if they don't exist
    os.makedirs(sim_path, exist_ok=True)
    os.makedirs(meas_path, exist_ok=True)
    if not glob.glob(os.path.join(sim_path, '*.mat')):
        print("Creating dummy .mat files for testing...")
        from scipy.io import savemat
        dummy_sim_data = np.random.rand(1000, 1) + 1j * np.random.rand(1000, 1)
        dummy_meas_data = np.random.rand(500, 1)
        for cls in base_cls + novel_cls:
             is_sim = cls in ['BTR', 'T72', 'F22'] # Example check
             path = sim_path if is_sim else meas_path
             var_name = 'CoHH' if is_sim else 'data'
             data_to_save = dummy_sim_data if is_sim else dummy_meas_data
             for i in range(5): # Create 5 dummy files per class
                 savemat(os.path.join(path, f"{cls}_dummy_{i}_test.mat"), {var_name: data_to_save})
        print("Dummy files created.")


    print("--- Testing Base Split ---")
    base_dataset = HRRPDataset(root_dirs=[sim_path, meas_path], split='base', classes=base_cls,
                               target_length=target_len, normalization=norm, phase_info='magnitude')
    if len(base_dataset) > 0:
        sample, label = base_dataset[0]
        print(f"Sample shape: {sample.shape}, Label: {label} ({base_dataset.idx_to_class[label]})")
        print(f"Min: {sample.min()}, Max: {sample.max()}")
    else:
        print("Base dataset is empty.")

    print("\n--- Testing Novel Split ---")
    novel_dataset = HRRPDataset(root_dirs=[sim_path, meas_path], split='novel', classes=novel_cls,
                                target_length=target_len, normalization=norm, phase_info='magnitude')
    if len(novel_dataset) > 0:
        sample, label = novel_dataset[0]
        print(f"Sample shape: {sample.shape}, Label: {label} ({novel_dataset.idx_to_class[label]})")
    else:
        print("Novel dataset is empty.")

    print("\n--- Testing Complex Data Handling ---")
    complex_dataset = HRRPDataset(root_dirs=[sim_path], split='base', classes=['BTR'], # Assuming BTR is simulated
                                  target_length=target_len, normalization='max', phase_info='complex')
    if len(complex_dataset) > 0:
        sample, label = complex_dataset[0]
        print(f"Complex sample shape: {sample.shape}") # Expected (2, target_len) if stacked real/imag
                                                        # Or (1, target_len) complex - check _preprocess_hrrp logic
    else:
        print("Complex dataset test skipped (no suitable data).")