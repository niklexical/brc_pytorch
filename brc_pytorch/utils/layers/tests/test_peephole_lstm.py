"""Testing peephole_lstm"""
import torch
from brc_pytorch.utils.layers.peephole_lstm import PeepholeLSTMCell


def test_lstm_cell():

    batch_size = 5
    input_size = 2
    hidden_size = 20

    input_set = torch.rand((batch_size, input_size))

    for bias in [True, False]:

        lstm = PeepholeLSTMCell(input_size, hidden_size, bias)
        hidden_state, cell_state = lstm.init_hidden(batch_size)

        assert type(input_set) == torch.Tensor

        hidden_state, cell_state = lstm(input_set, hidden_state, cell_state)

        assert hidden_state.size() == torch.Size([batch_size, hidden_size])
        assert cell_state.size() == torch.Size([batch_size, hidden_size])

        assert lstm.weights_c.size() == torch.Size(
            [hidden_size, hidden_size * 3]
        )
        assert lstm.weights_h.size() == torch.Size(
            [hidden_size, hidden_size * 4]
        )
        assert lstm.weights_x.size() == torch.Size(
            [input_size, hidden_size * 4]
        )
        assert lstm.bias.size() == torch.Size([hidden_size * 4])
