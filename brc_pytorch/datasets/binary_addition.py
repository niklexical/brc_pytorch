from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryAddition(Dataset):
    """Generates a dataset for binary addition."""

    def __init__(
        self, sample_size: int, sequence_length: int, max_sequence_length: int,
        min_sequence_length: int, mode: str
    ) -> None:
        """Constructor.

        Args:
            sample_size (int): Number of samples to generate.
            sequence_length (int): The sum of the lengths of the 2 binary
                numbers to add. Set it to 0 for mode = 'mixed'.
            max_sequence_length (int): Upper limit on length of the sequence.
            min_sequence_length (int): Lower limit length of the sequence.
            mode (string): 'single' generates a dataset for the given
                sequence length. 'mixed' generates a dataset containing all
                sequence lengths.
        """

        self.sample_size = sample_size
        self.max_seq_len = max_sequence_length
        self.min_seq_len = min_sequence_length
        self.sequence_length = sequence_length
        self.mode = mode
        self.x, self.y = self.create_dataset()

    def get_binary(self, length: int) -> str:
        """Generates a binary sequence.

        Args:
            length (int): Desired length of the binary sequence.

        Returns:
            string: binary sequence of specified length
        """
        return '{:0' + str(length) + 'b}'

    def create_dataset(self) -> Tuple:
        """Generates a dataset of binary numbers and their sums.

        Returns:
            Tuple: The input numbers and their corresponding sums are returned
                in binary integer form as a tuple.
        """

        if self.mode == 'single':
            # Input samples
            inputs = np.full(
                (self.sample_size, self.sequence_length + 1, 2), 0
            )
            # Target samples
            sums = np.full((self.sample_size, self.sequence_length + 1, 1), 0)

            for i in range(self.sample_size):

                max_len_nb1 = np.random.randint(1, self.sequence_length)
                max_len_nb2 = self.sequence_length - max_len_nb1

                # Generate random numbers to add
                nb1 = np.random.randint(2**(max_len_nb1 - 1), 2**(max_len_nb1))
                nb2 = np.random.randint(2**(max_len_nb2 - 1), 2**(max_len_nb2))
                sum_ = '{:0b}'.format(nb1 + nb2)

                inputs[i, :max_len_nb1, 0] = list(
                    reversed(
                        [
                            int(b)
                            for b in self.get_binary(max_len_nb1).format(nb1)
                        ]
                    )
                )
                inputs[i, :max_len_nb2, 1] = list(
                    reversed(
                        [
                            int(b)
                            for b in self.get_binary(max_len_nb2).format(nb2)
                        ]
                    )
                )
                sums[i, :len(sum_), 0] = list(reversed([int(b) for b in sum_]))

            return inputs, sums

        elif self.mode == 'mixed':
            sample_size = self.sample_size * (
                self.max_seq_len + 1 - self.min_seq_len
            )
            # Input samples
            inputs = np.full((sample_size, self.max_seq_len + 1, 2), 2)
            # Target samples
            sums = np.full((sample_size, self.max_seq_len + 1, 1), 2)

            for i in range(self.min_seq_len, self.max_seq_len + 1):
                for n in range(self.sample_size):
                    j = n + self.sample_size * (i - self.min_seq_len)
                    max_len_nb1 = np.random.randint(1, i)
                    max_len_nb2 = i - max_len_nb1

                    # Generate random numbers to add
                    nb1 = np.random.randint(
                        2**(max_len_nb1 - 1), 2**(max_len_nb1)
                    )
                    nb2 = np.random.randint(
                        2**(max_len_nb2 - 1), 2**(max_len_nb2)
                    )

                    sum_ = '{:0b}'.format(nb1 + nb2)

                    inputs[j, :max_len_nb1, 0] = list(
                        reversed(
                            [
                                int(b) for b in self.get_binary(max_len_nb1
                                                                ).format(nb1)
                            ]
                        )
                    )
                    inputs[j, :max_len_nb2, 1] = list(
                        reversed(
                            [
                                int(b) for b in self.get_binary(max_len_nb2
                                                                ).format(nb2)
                            ]
                        )
                    )
                    sums[j, :len(sum_), 0] = list(
                        reversed([int(b) for b in sum_])
                    )

            return inputs, sums

    def __len__(self) -> int:
        """Length of the data.

        Returns:
            int: Length of the data.
        """
        if self.mode == 'single':
            return self.sample_size
        elif self.mode == 'mixed':
            return len(self.x)

    def __getitem__(self, idx) -> Tuple:
        """Generates a subset of data.

        Args:
            idx (tensor): Indices to be sampled.

        Returns:
            x (tensor): Tensor of input binary numbers subset idx.
            y (tensor): Tensor of corresponding sums subset by idx.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.from_numpy(self.x[idx, :, :]).float()
        y = torch.from_numpy(self.y[idx, :, :]).float()

        return x, y
