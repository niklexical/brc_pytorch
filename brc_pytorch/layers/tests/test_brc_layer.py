"""Testing Bistable Recurrent Cell"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brc_pytorch.datasets import BRCDataset
from brc_pytorch.layers import BistableRecurrentCell, MultiLayerBase


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


def test_brc_cell():

    batch_size = 5
    input_size = 2
    hidden_size = 20

    input_set = torch.rand((batch_size, input_size))

    brc = BistableRecurrentCell(input_size, hidden_size)
    hidden_state = brc.get_initial_state(batch_size)

    hidden_state = brc(input_set, hidden_state)

    assert brc.memoryz.size() == torch.Size([hidden_size])
    assert brc.memoryz.size() == brc.memoryr.size()
    assert brc.kernelz.size() == torch.Size([input_size, hidden_size])
    assert brc.kernelr.size() == brc.kernelz.size()
    assert brc.kernelh.size() == brc.kernelz.size()

    assert brc.bz.size() == torch.Size([hidden_size])
    assert brc.br.size() == brc.bz.size()

    assert hidden_state.size() == torch.Size([batch_size, hidden_size])


def test_grad_flow(generate_dataset):

    input_size = 1
    hidden_sizes = [input_size, 100, 100]

    recurrent_layers = [
        BistableRecurrentCell(hidden_sizes[i], hidden_sizes[i + 1])
        for i in range(len(hidden_sizes) - 1)
    ]

    rnn = MultiLayerBase('BRC', recurrent_layers, hidden_sizes[1:])

    model = nn.Sequential(rnn, nn.Linear(hidden_sizes[2], 1))

    loss_fn_model = nn.MSELoss()

    optimiser_model = torch.optim.Adam(model.parameters())

    epochs = 20

    training_loader, inputs_test, outputs_test = generate_dataset

    for e in range(epochs):

        model.train()

        for idx, (x_batch, y_batch) in enumerate(training_loader):

            x_batch, y_batch = x_batch, y_batch
            pred_train = model(x_batch)

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

    assert len(param_groups_model) == 16
    assert sum(numweights_model) == 3 * input_size * hidden_sizes[
        1] + 4 * hidden_sizes[1] + 3 * hidden_sizes[1] * hidden_sizes[
            2] + 4 * hidden_sizes[2] + hidden_sizes[2] + 1
