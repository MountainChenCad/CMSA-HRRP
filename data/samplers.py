import torch
import numpy as np
import logging # Added

logger = logging.getLogger(__name__) # Added

class CategoriesSampler():
    """
    Samples batches for N-way K-shot tasks.
    Ensures each batch contains examples from N classes,
    with K support samples and Q query samples per class.
    """
    def __init__(self, label, n_batch, n_cls, n_per):
        """
        Args:
            label (list or np.ndarray): List of integer labels for the entire dataset (0 to C-1).
            n_batch (int): Number of batches (episodes) to generate.
            n_cls (int): Number of classes per batch (N-way).
            n_per (int): Number of samples per class in the batch (K-shot + Q-query).
        """
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = [] # Stores indices for each class label value
        # Ensure unique_labels are sorted integers corresponding to dataset internal labels
        self.unique_labels = sorted(list(set(label)))

        for i in self.unique_labels:
            ind = np.argwhere(label == i).reshape(-1)
            # Check if enough samples exist for this class
            if len(ind) < self.n_per:
                logger.warning(f"Class label {i} has only {len(ind)} samples, but {self.n_per} are needed per episode. "
                               f"Sampling for this class will use replacement or fail if insufficient.")
                # Decide on behavior: error out, or allow sampling with replacement?
                # For now, allow it and let the sampling logic handle it (or potentially fail later)
                # if len(ind) < 1: # Cannot sample even 1
                #      raise ValueError(f"Class label {i} has 0 samples. Cannot create episodes.")

            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        self.num_classes = len(self.unique_labels)
        if self.num_classes < self.n_cls:
             raise ValueError(f"Number of unique classes found in labels ({self.num_classes}) is less than n_cls ({self.n_cls})")
        logger.info(f"CategoriesSampler initialized: {self.num_classes} classes found. Sampling {self.n_batch} episodes.")


    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            # Randomly select N classes *indices* from the available classes (0 to num_classes-1)
            selected_class_indices = torch.randperm(self.num_classes)[:self.n_cls]

            for class_idx in selected_class_indices:
                indices_for_class = self.m_ind[class_idx] # Get all indices for this class label
                num_samples_in_class = len(indices_for_class)

                if num_samples_in_class < self.n_per:
                    # Sample with replacement if not enough samples
                    logger.debug(f"Sampling class index {class_idx} with replacement ({num_samples_in_class} < {self.n_per})")
                    pos = torch.randint(0, num_samples_in_class, (self.n_per,))
                else:
                    # Sample without replacement
                    pos = torch.randperm(num_samples_in_class)[:self.n_per]

                batch.append(indices_for_class[pos])

            # Concatenate indices for the batch
            # Shape was (N, K+Q), transpose to (K+Q, N), reshape to (N*(K+Q))
            # The order will be: [cls1_samp1, cls2_samp1, ..., clsN_samp1, cls1_samp2, ...]
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

# --- Other Samplers (Optional, kept from original SemFew code - UNUSED in current pipeline) ---
# ... (RandomSampler, ClassSampler, InSetSampler remain unchanged) ...