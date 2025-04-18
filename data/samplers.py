import torch
import numpy as np

class CategoriesSampler():
    """
    Samples batches for N-way K-shot tasks.
    Ensures each batch contains examples from N classes,
    with K support samples and Q query samples per class.
    (Code reused from SemFew / standard FSL implementations)
    """
    def __init__(self, label, n_batch, n_cls, n_per):
        """
        Args:
            label (list or np.ndarray): List of integer labels for the entire dataset.
            n_batch (int): Number of batches (episodes) to generate.
            n_cls (int): Number of classes per batch (N-way).
            n_per (int): Number of samples per class in the batch (K-shot + Q-query).
        """
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = [] # Stores indices for each class
        unique_labels = sorted(list(set(label)))

        for i in unique_labels:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        self.num_classes = len(self.m_ind)
        if self.num_classes < self.n_cls:
             raise ValueError(f"Number of classes in dataset ({self.num_classes}) is less than n_cls ({self.n_cls})")


    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            # Randomly select N classes
            classes = torch.randperm(self.num_classes)[:self.n_cls]
            for c in classes:
                l = self.m_ind[c] # Get indices for class c
                # Randomly select K+Q samples from the chosen class
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])

            # Concatenate indices for the batch
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transposes, making shape [n_per, n_cls]
            # .reshape(-1) flattens to [n_per * n_cls]
            yield batch

# --- Other Samplers (Optional, kept from original SemFew code) ---

class RandomSampler():

    def __init__(self, label, n_batch, n_per):
        self.n_batch = n_batch
        self.n_per = n_per
        self.label = np.array(label)
        self.num_label = self.label.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = torch.randperm(self.num_label)[:self.n_per]
            yield batch


# sample for each class
class ClassSampler():

    def __init__(self, label, n_per=None):
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        unique_labels = sorted(list(set(label)))
        for i in unique_labels:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return len(self.m_ind)

    def __iter__(self):
        classes = torch.arange(len(self.m_ind))
        for c in classes:
            l = self.m_ind[int(c)]
            if self.n_per is None:
                pos = torch.randperm(len(l))
            else:
                pos = torch.randperm(len(l))[:self.n_per]
            yield l[pos]


# for ResNet Fine-Tune, which output the same index of task examples several times
class InSetSampler():

    def __init__(self, n_batch, n_sbatch, pool): # pool is a tensor
        self.n_batch = n_batch
        self.n_sbatch = n_sbatch
        self.pool = pool
        self.pool_size = pool.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = self.pool[torch.randperm(self.pool_size)[:self.n_sbatch]]
            yield batch