import math
import torch
import torch.nn as nn
from typing import Tuple


class PeepholeLSTMCell(nn.Module):
    """LSTM with peephole connections."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        *args,
        **kwargs
    ) -> None:
        """Constructor.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (dict): Number of hidden units in linear and
                encoder layers.
            bias (bool): Whether to include bias. Defaults to True.
        """
        super(PeepholeLSTMCell, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weights_x = nn.Parameter(
            torch.FloatTensor(self.input_size, self.hidden_size * 4)
        )
        self.weights_h = nn.Parameter(
            torch.FloatTensor(self.hidden_size, self.hidden_size * 4)
        )
        self.weights_c = nn.Parameter(
            torch.FloatTensor(self.hidden_size, self.hidden_size * 3)
        )

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.hidden_size * 4))
        else:
            self.bias = torch.zeros((self.hidden_size * 4))

        self.init_params()

    def init_params(self) -> None:
        """Uniform Xavier initialisation of weights."""

        std_dev = math.sqrt(1 / self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std_dev, std_dev)

    def init_hidden(self, batch_size: int) -> Tuple:
        """Initialise hidden and cell states.

        Args:
            batch_size (int) : batch size used for training.

        Returns:
            Tuple: a tuple containing the hidden state and cell state
                both initialized at 0s.
        """
        hidden_state = torch.FloatTensor(
            torch.zeros((batch_size, self.hidden_size))
        )
        cell_state = torch.FloatTensor(
            torch.zeros((batch_size, self.hidden_size))
        )

        return hidden_state, cell_state

    def forward(
        self, data_t: torch.Tensor, hidden_state: torch.Tensor,
        cell_state: torch.Tensor
    ) -> Tuple:
        """Single LSTM cell.

        Args:
            data_t (torch.Tensor): Element at step t with shape
                [batch_size, self.input_size]
            hidden_state (torch.Tensor): Hidden state of the LSTM cell of shape
                [batch_size, self.hidden_size]
            cell_state (torch.Tensor): Cell state of the LSTM cell of shape
                [batch_size, self.hidden_size]

        Returns:
            Tuple: A tuple containing the hidden state and cell state
                after the element is processed.
        """

        linear_xh = torch.matmul(data_t, self.weights_x) + torch.matmul(
            hidden_state, self.weights_h
        ) + self.bias

        linear_cxh = linear_xh[:, :self.hidden_size * 2] + torch.matmul(
            cell_state, self.weights_c[:, :self.hidden_size * 2]
        )

        forget_prob = torch.sigmoid(linear_cxh[:, :self.hidden_size])

        input_prob = torch.sigmoid(
            linear_cxh[:, self.hidden_size:self.hidden_size * 2]
        )

        candidates = torch.tanh(
            linear_xh[:, self.hidden_size * 2:self.hidden_size * 3]
        )

        cell_state = forget_prob * cell_state + input_prob * candidates

        linear_output = (
            linear_xh[:, self.hidden_size * 3:self.hidden_size * 4] +
            torch.matmul(
                cell_state,
                self.weights_c[:, self.hidden_size * 2:self.hidden_size * 3]
            )
        )

        output_gate = torch.sigmoid(linear_output)

        hidden_state = torch.tanh(cell_state) * output_gate

        return hidden_state, cell_state
