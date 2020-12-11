import torch.nn as nn


class SelectItem(nn.Module):
    """Select output from a tuple to pass onto next function."""

    def __init__(
        self, item_index: int, elem_idx: int = None, batch_first: bool = True
    ) -> None:
        """Item selection module that can be included in nn.Sequential().

        Args:
            item_index (int): Index of the item to retrieve.
            elem_idx (int, optional): Index from which the selected item should
                be sliced. Depending on whether batch_first is True or False, the
                dimension along which it is sliced is either 0 or 1. Also,
                this implementation slices from the elem_idx until the end. E.g
                if elem_idx = -5, then the last 5 elements from the item are
                retained. Default: None.
            batch_first (bool, optional): Used as an indicator to slice along the
                correct dimensions using elem_idx.
        """
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index
        self.elem_idx = elem_idx
        self.batch_first = batch_first

    def forward(self, inputs):
        """Selects item from tuple/list, slices it and returns it.

        Args:
            inputs (Tuple/List): Tuple/List from which item is to be retrieved.

        Returns:
            Tensor/Array: Selected and sliced (if elem_idx is given) Tensor/Array.
        """
        if self.elem_idx:
            if self.batch_first:
                return inputs[self.item_index][:, self.elem_idx:, :]
            else:
                return inputs[self.item_index][self.elem_idx:, :, :]
        else:
            return inputs[self.item_index]
