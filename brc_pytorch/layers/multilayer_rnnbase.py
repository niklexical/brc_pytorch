from typing import Tuple

import torch
import torch.nn as nn


class MultiLayerBase(nn.Module):
    """Multi layer forward recurrent network."""

    def __init__(
        self,
        mode,
        cells,
        hidden_sizes: int,
        device: str = 'cpu',
        return_sequences: bool = False,
    ) -> None:
        """Constructor.

        Args:
            mode (string): Type of cell to be used in RNN. Options are:
                ['GRU', 'LSTM', 'RNN', 'BRC', 'nBRC'].
            cell (list): List of cells to be used in the RNN layers.
            hidden_sizes (int): Hidden sizes for each cell.
            device (str): Whether to run model on GPU or CPU. Defaults to CPU.
            return_sequences (bool): Return the hidden states for all
                time steps. Defaults to False.
        """
        super().__init__()
        self.mode = mode
        self.cells = nn.ModuleList(cells)
        self.hidden_sizes = hidden_sizes
        self.return_seq = return_sequences
        self.device = device

    def set_hidden_state(
        self,
        hidden_state: torch.FloatTensor = None,
        batch_size: int = None,
        hidden_size: int = None
    ) -> None:
        """Set the hidden state for current cell.

        Args:
            hidden_state (torch.FloatTensor, optional): Hidden state to be used
                in computation of the next hidden state. Defaults to None.
            batch_size (int, optional): Batch size of hidden state initialised
                to 0s if a new one should be created. Defaults to None.
            hidden_size (int, optional): Dimension of hidden state initialised
                to 0s if a new one should be created. Defaults to None.
        """

        if hidden_state is not None:
            self.hidden_state = hidden_state.to(self.device)
        else:
            self.hidden_state = torch.zeros(
                (batch_size, hidden_size), dtype=torch.float32
            ).to(self.device)

    def set_cell_state(
        self,
        cell_state: torch.FloatTensor = None,
        batch_size: int = None,
        hidden_size: int = None
    ) -> None:
        """Set the cell state for current cell.

        Args:
            cell_state (torch.FloatTensor, optional): Cell state to be used
                in computation of the next cell and hidden states.
                Defaults to None.
            batch_size (int, optional): Batch size of cell state initialised
                to 0s if a new one should be created. Defaults to None.
            hidden_size (int, optional): Dimension of cell state initialised
                to 0s if a new one should be created. Defaults to None.
        """

        if cell_state is not None:
            self.cell_state = cell_state.to(self.device)
        else:
            self.cell_state = torch.zeros(
                (batch_size, hidden_size), dtype=torch.float32
            ).to(self.device)

    def multilayer_rnn(self, inputs: torch.Tensor) -> Tuple:
        """Multi layer implementation of RNN/GRU cells.

        Args:
            inputs (torch.Tensor): Input tensor of shape
                [batch size, sequence length, input size].

        Returns:
            Tuple: If return sequence is True, the hidden states from all time
                steps of the last layer and from all layers are returned with
                shapes [batch_size,sequence_length,hidden_size] and
                [batch_size, sequence_length, sum(self.hidden_sizes)]
                respectively. Otherwise, returns the last hidden state from
                the last layer with shape [batch_size,hidden_size].
        """
        batch_size, sequence_length, input_size = inputs.size()

        all_hidden_states = []

        for c in range(len(self.cells)):
            stacked_states = []
            self.set_hidden_state(None, batch_size, self.hidden_sizes[c])

            for i in range(sequence_length):

                if c == 0:
                    inp = inputs[:, i, :]
                else:
                    inp = previous_cell_states[:, i, :].to(self.device)

                new_state = self.cells[c](inp, self.hidden_state)

                self.set_hidden_state(new_state)

                stacked_states.append(new_state)

            previous_cell_states = torch.stack(stacked_states).permute(1, 0, 2)

            if self.return_seq:
                all_hidden_states.append(previous_cell_states)

        if self.return_seq:
            return previous_cell_states, torch.cat(all_hidden_states, 2)
        else:
            return new_state

    def multilayer_lstm(self, inputs: torch.Tensor) -> Tuple:
        """Multi layer implementation of LSTM cells.

        Args:
            inputs (torch.Tensor): Input tensor of shape
                [batch size, sequence length, input size].

        Returns:
            Tuple: If return sequence is True, the hidden and cell states
                from all time steps of the last layer and from all layers are
                returned with shapes [batch_size,sequence_length,hidden_size]
                and [batch_size, sequence_length, sum(self.hidden_sizes)]
                respectively. Otherwise, returns the last hidden and cell state
                from the last layer with shape [batch_size,hidden_size].
        """
        batch_size, sequence_length, input_size = inputs.size()

        all_hidden_states = []
        all_cell_states = []

        for c in range(len(self.cells)):
            stacked_hidden_states = []
            stacked_cell_states = []

            self.set_hidden_state(None, batch_size, self.hidden_sizes[c])
            self.set_cell_state(None, batch_size, self.hidden_sizes[c])

            for i in range(sequence_length):

                if c == 0:
                    inp = inputs[:, i, :]

                else:
                    inp = previous_hidden_states[:, i, :].to(self.device)

                new_hidden_state, new_cell_state = self.cells[c](
                    inp, (self.hidden_state, self.cell_state)
                )

                self.set_hidden_state(new_hidden_state)
                self.set_cell_state(new_cell_state)

                stacked_hidden_states.append(new_hidden_state)
                stacked_cell_states.append(new_cell_state)

            previous_hidden_states = torch.stack(stacked_hidden_states
                                                 ).permute(1, 0, 2)

            previous_cell_states = torch.stack(stacked_cell_states
                                               ).permute(1, 0, 2)

            if self.return_seq:
                all_hidden_states.append(previous_hidden_states)
                all_cell_states.append(previous_cell_states)

        if self.return_seq:
            return previous_hidden_states, previous_cell_states, torch.cat(
                all_hidden_states, 2
            ), torch.cat(all_cell_states, 2)
        else:
            return new_hidden_state, new_cell_state

    def forward(self, inputs: torch.Tensor) -> Tuple:

        if self.mode in ['GRU', 'RNN', 'BRC', 'nBRC']:
            return self.multilayer_rnn(inputs)

        elif self.mode == 'LSTM':
            return self.multilayer_lstm(inputs)
