import numpy as np
import torch
from torch.utils.data import Dataset


class BRCDataset(Dataset):
    """Dataset of sets."""

    def __init__(
        self,
        inputs: np.array,
        outputs: np.array,
    ) -> None:
        """Constructor.

        Args:
            inputs (np.array): Numpy array of input data.
            outputs (np.array): Numpy array of output data
        """
        self.input = inputs
        self.output = outputs

    def __len__(self) -> int:
        """Length of the data.

        Returns:
            int: Length of the data.
        """
        return len(self.input)

    def __getitem__(self, idx):
        """Generates a subset of data.

        Args:
            idx (tensor): Indices to be sampled.

        Returns:
            x (tensor): Tensor of input data.
            y (tensor): Tensor of corresponding outputs.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.from_numpy(self.input[idx, :])
        y = torch.from_numpy(self.output[idx, :])

        return x, y
