import torch.nn as nn


class SelectItem(nn.Module):
    """Select output from a tuple to pass onto next function."""

    def __init__(self, item_index: int) -> None:
        """Constructor.

        Args:
            item_index (int): Index of the item to retrieve.
        """
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        """Selects item from tuple/list and returns it.

        Args:
            inputs (Tuple/List): Tuple/List from which item is to be retrieved.

        Returns:
            Tensor/Array: Tensor/Array corresponding to item_index in the inputs.
        """
        return inputs[self.item_index]
