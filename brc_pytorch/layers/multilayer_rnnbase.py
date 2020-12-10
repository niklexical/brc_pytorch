from typing import Tuple, List
import copy
import torch
import torch.nn as nn


class MultiLayerBase(nn.Module):
    """Executes a multilayer recurrent network given RNN cells for each layer."""

    def __init__(
        self,
        mode: str,
        cells: List[torch.nn.Module],
        hidden_size: int,
        batch_first: bool = True,
        bidirectional: bool = False,
        return_sequences: bool = False,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
    ) -> None:
        """Constructor.

        Args:
            mode (string): Type of cell to be used in the recurrent network.
                Options are: ['GRU', 'LSTM', 'RNN', 'BRC', 'nBRC'].
            cell (List[torch.nn.Module]): List of cells to be used in each
            network layer. For example, torch.nn.GRUCell module initialised with
            the necessary arguments repeated num_layers times, i.e,
            [torch.nn.GRUCell(8,16),torch.nn.GRUCell(16,16)] for a 2 layered
            network with input_size = 8 and hidden_size = 16.
            NOTE: For bidirectional=True, remember to double the input_size for
            cells of layer 2 onwards, i.e, borrowing from the example above,
            [torch.nn.GRUCell(8,16),torch.nn.GRUCell(16*2,16)].
            hidden_size (int): Size of the hidden state.
            batch_first (bool): If True, then the input and output tensors are
                provided as (batch, seq_len, input_size). Default: False.
            bidirectional (bool): If True, becomes a bidirectional network, i.e,
                input is fed into 2 independent recurrent networks in the forward
                and backward direction. Default: False.
            return_sequences (bool): Return the hidden states for all
                time steps. Defaults to False.
            device (str): Whether to run model on GPU or CPU. Defaults to 'cuda',
            if available.
        """
        super().__init__()

        self.mode = mode

        self.layers = len(cells)
        self.hidden_size = hidden_size
        self.cells_forward = cells

        self.batch_first = batch_first
        self.return_seq = return_sequences
        self.device = device

        self.bidirectional = bidirectional

        self.num_directions = 2 if self.bidirectional else 1

        if self.bidirectional:
            self.cells_backward = copy.deepcopy(self.cells_forward)
            self.cells = nn.ModuleList(
                [
                    x for z in zip(self.cells_forward, self.cells_backward)
                    for x in z
                ]
            )
            assert len(self.cells) == self.layers * 2
        else:
            self.cells = nn.ModuleList(self.cells_forward)

    def init_state(self, batch_size: int) -> torch.Tensor:
        """Initialises the hidden/cell state for the recurrent network.

        Args:
            batch_size (int): Batch size of the state tensor initialised
                to 0s, should a new one be created.

        Returns:
            torch.Tensor: Tensor of 0s with shape
                (layers*num_directions, batch, hidden_size).
        """
        state = torch.zeros(
            (self.layers * self.num_directions, batch_size, self.hidden_size),
            dtype=torch.float32,
        ).to(self.device)
        return state

    def set_hidden_state(self, hidden_state: torch.FloatTensor) -> None:
        """Sets the hidden state for current cell.

        Args:
            hidden_state (torch.FloatTensor): Hidden state to be used
                in computation of the next hidden state.
        """

        self.hidden_state = hidden_state.to(self.device)

    def set_cell_state(self, cell_state: torch.FloatTensor) -> None:
        """Set the cell state for current cell.

        Args:
            cell_state (torch.FloatTensor): Cell state to be used
                in computation of the next cell and hidden states.
        """

        self.cell_state = cell_state.to(self.device)

    def multilayer_rnn(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor = None,
    ) -> Tuple:
        """Multi-layer implementation of cells with only a hidden state.

        Args:
            inputs (torch.Tensor): Input tensor to be processed. If
                batch_first = True, (batch, seq_len, input_size) is
                expected, otherwise a tensor with shape
                (seq_len, batch, input_size) should be provided.
            NOTE: The operations in this function assume a 3-dimensional input.
                    For higher dimensions, please adapt the permutations
                    accordingly.
            state (torch.Tensor, optional): Tensor to intialise the hidden state
                with shape (layers*num_directions,batch,hidden_size).
                Default: None.
        Returns:
            Tuple: If return_sequence = True, returns an output tensor of shape
                (batch, seq_len, num_directions * hidden_size)
                for batch_first = True, and (seq_len, batch, num_directions * hidden_size)
                for batch_first = False, containing the output
                features h_t from the last layer of the network, for each t,
                as well as a tensor containing the hidden state for t = seq_len,
                with shape (layers * num_directions, batch, hidden_size).
                If return_sequence = False, returns the last hidden state from
                the last layer with shape (num_directions, batch, hidden_size).
        """
        if self.batch_first is False:
            inputs = inputs.permute(1, 0, 2)

        batch_size, sequence_length, input_size = inputs.size()

        last_hidden_states = []

        if state is None:
            hidden_state = self.init_state(batch_size)
        else:
            hidden_state = state

        for c in range(self.layers):
            layer_outputs = []

            for d in range(self.num_directions):

                stacked_states = []

                self.set_hidden_state(
                    hidden_state[self.num_directions * c + d]
                )

                range_length = (
                    range(sequence_length)
                    if d == 0 else reversed(range(sequence_length))
                )

                for i in range_length:

                    if c == 0:
                        inp = inputs[:, i, :]
                    else:
                        inp = output[:, i, :].to(self.device)

                    new_state = self.cells[self.num_directions * c +
                                           d](inp, self.hidden_state)

                    self.set_hidden_state(new_state)

                    stacked_states.append(new_state)

                layer_stacked = torch.stack(stacked_states).permute(1, 0, 2)

                last_hidden_states.append(layer_stacked[:, -1, :])

                if d == 1:
                    layer_stacked = layer_stacked.flip(1)

                layer_outputs.append(layer_stacked)

            output = torch.cat(layer_outputs, dim=-1)

        h_n = torch.stack(last_hidden_states)

        if self.batch_first is False:
            output = output.permute(1, 0, 2)

        if self.return_seq:
            return output, h_n

        else:
            return h_n[-2:, :, :
                       ] if self.bidirectional else h_n[-1, :, :].unsqueeze(0)

    def multilayer_lstm(self, inputs: torch.Tensor, states: Tuple) -> Tuple:
        """Multi-layer implementation of cells with a hidden and cell state.

        Args:
            inputs (torch.Tensor): Input tensor to be processed. If
                batch_first = True, (batch, seq_len, input_size) is
                expected, otherwise a tensor with shape
                (seq_len, batch, input_size) should be provided.
            NOTE: The operations in this function assume a 3-dimensional input.
                    For higher dimensions, please adapt the permutations
                    accordingly.
            state (Tuple, optional): Tuple of tensors to intialise the hidden
                and cell states with shape
                (layers * num_directions, batch, hidden_size). Default: None.
        Returns:
            Tuple: If return_sequence = True, returns an output tensor of shape
                (batch, seq_len, num_directions * hidden_size)
                for batch_first = True, and (seq_len, batch, num_directions * hidden_size)
                for batch_first = False, containing the output
                features h_t from the last layer of the network, for each t,
                as well as a tuple of tensors containing the hidden and cell
                states for t = seq_len, with each having a shape of
                (layers * num_directions, batch, hidden_size).
                If return_sequence = False, returns the hidden and cell state at
                t = seq_len from the last layer with shape
                (num_directions, batch, hidden_size).
        """

        if self.batch_first is False:
            inputs = inputs.permute(1, 0, 2)

        batch_size, sequence_length, input_size = inputs.size()

        last_hidden_states = []
        last_cell_states = []

        if states is None:
            hidden_state = self.init_state(batch_size)
            cell_state = self.init_state(batch_size)
        else:
            hidden_state, cell_state = states

        for c in range(self.layers):
            layer_hidden_outputs = []

            for d in range(self.num_directions):
                stacked_hidden_states = []
                stacked_cell_states = []

                self.set_hidden_state(
                    hidden_state[self.num_directions * c + d]
                )
                self.set_cell_state(cell_state[self.num_directions * c + d])

                range_length = (
                    range(sequence_length)
                    if d == 0 else reversed(range(sequence_length))
                )

                for i in range_length:

                    if c == 0:
                        inp = inputs[:, i, :]

                    else:
                        inp = output_hidden_states[:, i, :].to(self.device)

                    new_hidden_state, new_cell_state = self.cells[
                        self.num_directions * c +
                        d](inp, (self.hidden_state, self.cell_state))

                    self.set_hidden_state(new_hidden_state)
                    self.set_cell_state(new_cell_state)

                    stacked_hidden_states.append(new_hidden_state)
                    stacked_cell_states.append(new_cell_state)

                layer_hidden_stacked = torch.stack(stacked_hidden_states
                                                   ).permute(1, 0, 2)
                layer_cell_stacked = torch.stack(stacked_cell_states
                                                 ).permute(1, 0, 2)

                last_hidden_states.append(layer_hidden_stacked[:, -1, :])
                last_cell_states.append(layer_cell_stacked[:, -1, :])

                if d == 1:
                    layer_hidden_stacked = layer_hidden_stacked.flip(1)
                    layer_cell_stacked = layer_cell_stacked.flip(1)

                layer_hidden_outputs.append(layer_hidden_stacked)

            output_hidden_states = torch.cat(layer_hidden_outputs, dim=-1)

        h_n = torch.stack(last_hidden_states)
        c_n = torch.stack(last_cell_states)

        if self.batch_first is False:
            output_hidden_states = output_hidden_states.permute(1, 0, 2)

        if self.return_seq:
            return (
                output_hidden_states,
                (h_n, c_n),
            )
        else:

            return (
                (h_n[-2:, :, :], c_n[-2:, :, :]) if self.bidirectional else
                (h_n[-1, :, :].unsqueeze(0), c_n[-1, :, :].unsqueeze(0))
            )

    def forward(
        self, inputs: torch.Tensor, hidden_state: Tuple = None
    ) -> Tuple:
        """Executes the multilayer recurrent neural network based on the cell type.

        Args:
            inputs (torch.Tensor): Tensor to process.
            hidden_state (Tuple, optional): Tensor or tuple of tensor to initialise
                the recurrent cell states. Defaults to None.

        Returns:
            Tuple: The output returned by the network on processing the input.
            NOTE: In the bidrectional case, the hidden states at t=seq_len are
                simply the last returned hidden states in forward direction (t=n)
                in the reverse direction (t=0) for a given sequence with n steps.
                The output tensor is simply a concatenation of the hidden states
                at each time step t.
        """

        if self.mode in ['GRU', 'RNN', 'BRC', 'nBRC']:
            return self.multilayer_rnn(inputs, hidden_state)

        elif self.mode == 'LSTM':
            return self.multilayer_lstm(inputs, hidden_state)
