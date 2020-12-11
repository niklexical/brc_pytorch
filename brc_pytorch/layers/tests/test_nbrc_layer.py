"""Testing Neuromodulated Bistable Recurrent Cell"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brc_pytorch.datasets import BRCDataset
from brc_pytorch.layers import (
    MultiLayerBase, NeuromodulatedBistableRecurrentCell
)


@pytest.fixture
def generate_sample():
    n = 10
    true_n = np.random.randn()
    chain = np.concatenate([[true_n], np.random.randn(n - 1)])
    return chain, true_n


@pytest.fixture
def generate_dataset(generate_sample):

    # lag = 20
    test_size = 100
    inputs = []
    outputs = []

    for i in range(1000):
        # data = generate_sample()
        inp, out = generate_sample
        inputs.append(inp)
        outputs.append(out)

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    inputs_train = np.expand_dims(inputs, axis=2).astype(np.float32)

    inputs_test = np.expand_dims(inputs,
                                 axis=2).astype(np.float32)[-test_size:]

    outputs_train = np.expand_dims(outputs, axis=1).astype(np.float32)

    outputs_test = np.expand_dims(outputs,
                                  axis=1).astype(np.float32)[-test_size:]

    dataset_train = BRCDataset(inputs_train, outputs_train)

    training_loader = DataLoader(dataset_train, batch_size=100)

    return training_loader, inputs_test, outputs_test


def test_nbrc_cell():

    batch_size = 5
    input_size = 2
    hidden_size = 20

    input_set = torch.rand((batch_size, input_size))

    nbrc = NeuromodulatedBistableRecurrentCell(input_size, hidden_size)
    hidden_state = nbrc.get_initial_state(batch_size)

    hidden_state = nbrc(input_set, hidden_state)

    assert nbrc.memoryz.size() == torch.Size([hidden_size, hidden_size])
    assert nbrc.memoryz.size() == nbrc.memoryr.size()
    assert nbrc.kernelz.size() == torch.Size([input_size, hidden_size])
    assert nbrc.kernelr.size() == nbrc.kernelz.size()
    assert nbrc.kernelh.size() == nbrc.kernelz.size()

    assert nbrc.bz.size() == torch.Size([hidden_size])
    assert nbrc.br.size() == nbrc.bz.size()

    assert hidden_state.size() == torch.Size([batch_size, hidden_size])


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_grad_flow(generate_dataset, batch_first, bidirectional, num_layers):

    input_size = 1
    hidden_size = 100

    num_directions = 2 if bidirectional else 1

    recurrent_layers = [
        NeuromodulatedBistableRecurrentCell(input_size, hidden_size)
    ]

    inner_input_dimensions = num_directions * hidden_size

    for _ in range(num_layers - 1):
        recurrent_layers.append(
            NeuromodulatedBistableRecurrentCell(
                inner_input_dimensions, hidden_size
            )
        )

    rnn = MultiLayerBase(
        "nBRC",
        recurrent_layers,
        hidden_size,
        batch_first=batch_first,
        bidirectional=bidirectional,
        return_sequences=False,
    )

    fc = nn.Linear(num_directions * hidden_size, 1)

    model = nn.ModuleList([rnn, fc])
    loss_fn_model = nn.MSELoss()

    optimiser_model = torch.optim.Adam(model.parameters())

    epochs = 20

    training_loader, inputs_test, outputs_test = generate_dataset

    for e in range(epochs):

        model.train()

        for idx, (x_batch, y_batch) in enumerate(training_loader):
            # data generated always has batch first.
            batch_size = x_batch.size(0)

            if batch_first is False:
                # permuting data so that it follow batch_first = False scheme
                x_batch = x_batch.permute(1, 0, 2)

            pred_train = model[0](x_batch)

            # since output is (num_directions,batch,hidden_size), we first
            # permute it to get (batch,num_directions,hidden_size) and then
            # reshape it to (batch, num_directions * hidden_size) so it can be
            # fed into the FC layer

            pred_train = pred_train.permute(1, 0, 2)

            assert pred_train.size() == torch.Size(
                [batch_size, num_directions, hidden_size]
            )

            pred_train = pred_train.reshape(
                batch_size, num_directions * hidden_size
            )
            pred_train = model[1](pred_train)

            train_loss = loss_fn_model(pred_train, y_batch)

            optimiser_model.zero_grad()
            train_loss.backward()
            optimiser_model.step()

    param_groups_model = []
    numweights_model = []
    for n, p in model.named_parameters():
        param_groups_model.append(n)
        assert p.grad is not None

        numweights_model.append(torch.numel(p))

    # add 2 to account for linear layer parameter groups
    assert len(param_groups_model) == num_layers * 7 * num_directions + 2

    assert sum(numweights_model) == (
        (3 * input_size * hidden_size + 2 * hidden_size**2 + 2 * hidden_size) *
        num_directions + (
            3 * hidden_size * num_directions * hidden_size +
            2 * hidden_size**2 + 2 * hidden_size
        ) * (num_layers - 1) * num_directions +
        (num_directions * hidden_size) + 1
    ), f'{numweights_model}'
