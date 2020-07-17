"""Test Multilayer implementation by comparing it with in-built functions"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brc_pytorch.datasets import BRCDataset
from brc_pytorch.layers.multilayer_rnnbase import MultiLayerBase


class SelectItem(nn.Module):

    def __init__(self, item_index, model='custom'):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index
        self.model = model

    def forward(self, inputs):
        if self.model == 'torch':
            return inputs[self.item_index][:, -1, :]
        else:
            return inputs[self.item_index]


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


def test_multilayer_rnn(generate_dataset):
    """Tests multilayer functionality for cells that output hidden state only."""

    input_size = 1
    hidden_sizes = [input_size, 16, 16]

    recurrent_layers = [
        nn.GRUCell(hidden_sizes[i], hidden_sizes[i + 1])
        for i in range(len(hidden_sizes) - 1)
    ]

    rnn = MultiLayerBase('GRU', recurrent_layers, hidden_sizes[1:])

    model = nn.Sequential(rnn, nn.Linear(hidden_sizes[2], 1))

    model_torch = nn.Sequential(
        nn.GRU(input_size, 100, 2, batch_first=True), SelectItem(0, 'torch'),
        nn.Linear(hidden_sizes[2], 1)
    )

    loss_fn_model = nn.MSELoss()
    loss_fn_torch = nn.MSELoss()
    optimiser_model = torch.optim.Adam(model.parameters())
    optimiser_torch = torch.optim.Adam(model_torch.parameters())

    epochs = 20

    training_loader, inputs_test, outputs_test = generate_dataset
    test_x = torch.from_numpy(inputs_test)
    test_y = torch.from_numpy(outputs_test)

    for e in range(epochs):

        model.train()
        model_torch.train()

        for idx, (x_batch, y_batch) in enumerate(training_loader):

            x_batch, y_batch = x_batch, y_batch
            pred_train = model(x_batch)
            pred_train_torch = model_torch(x_batch)
            train_loss = loss_fn_model(pred_train, y_batch)
            train_loss_torch = loss_fn_torch(pred_train_torch, y_batch)

            optimiser_model.zero_grad()
            train_loss.backward()
            optimiser_model.step()

            optimiser_torch.zero_grad()
            train_loss_torch.backward()
            optimiser_torch.step()

    model.eval()
    my_test_pred = model(test_x)
    my_test_loss = loss_fn_model(my_test_pred, test_y)

    model_torch.eval()
    torch_test_pred = model_torch(test_x)
    torch_test_loss = loss_fn_torch(torch_test_pred, test_y)

    param_groups_model = []
    numweights_model = []
    for n, p in model.named_parameters():
        param_groups_model.append(n)
        assert p.grad is not None

        numweights_model.append(torch.numel(p))

    param_groups_torch = []
    numweights_torch = []
    for nt, pt in model_torch.named_parameters():
        param_groups_torch.append(nt)
        assert pt.grad is not None

        numweights_torch.append(torch.numel(pt))

    assert numweights_model == numweights_torch
    assert len(param_groups_model) == 10
    assert len(param_groups_torch) == 10
    assert x_batch.size() == torch.Size([100, 10, 1])
    assert torch.allclose(train_loss, train_loss_torch)
    assert torch.allclose(my_test_loss, torch_test_loss)


def test_multilayer_lstm(generate_dataset):
    """Tests multilayer functionality for cells that output hidden and cell state."""

    input_size = 1
    hidden_sizes = [input_size, 16, 16]

    recurrent_layers = [
        nn.LSTMCell(hidden_sizes[i], hidden_sizes[i + 1])
        for i in range(len(hidden_sizes) - 1)
    ]

    rnn = MultiLayerBase('LSTM', recurrent_layers, hidden_sizes[1:])

    model = nn.Sequential(rnn, SelectItem(0), nn.Linear(hidden_sizes[2], 1))

    model_torch = nn.Sequential(
        nn.LSTM(input_size, 100, 2, batch_first=True), SelectItem(0, 'torch'),
        nn.Linear(hidden_sizes[2], 1)
    )

    loss_fn_model = nn.MSELoss()
    loss_fn_torch = nn.MSELoss()
    optimiser_model = torch.optim.Adam(model.parameters())
    optimiser_torch = torch.optim.Adam(model_torch.parameters())

    epochs = 20

    training_loader, inputs_test, outputs_test = generate_dataset
    test_x = torch.from_numpy(inputs_test)
    test_y = torch.from_numpy(outputs_test)

    for e in range(epochs):

        model.train()
        model_torch.train()

        for idx, (x_batch, y_batch) in enumerate(training_loader):

            x_batch, y_batch = x_batch, y_batch
            pred_train = model(x_batch)
            pred_train_torch = model_torch(x_batch)
            train_loss = loss_fn_model(pred_train, y_batch)
            train_loss_torch = loss_fn_torch(pred_train_torch, y_batch)

            optimiser_model.zero_grad()
            train_loss.backward()
            optimiser_model.step()

            optimiser_torch.zero_grad()
            train_loss_torch.backward()
            optimiser_torch.step()

    model.eval()
    my_test_pred = model(test_x)
    my_test_loss = loss_fn_model(my_test_pred, test_y)

    model_torch.eval()
    torch_test_pred = model_torch(test_x)
    torch_test_loss = loss_fn_torch(torch_test_pred, test_y)

    param_groups_model = []
    numweights_model = []
    for n, p in model.named_parameters():
        param_groups_model.append(n)
        assert p.grad is not None

        numweights_model.append(torch.numel(p))

    param_groups_torch = []
    numweights_torch = []
    for nt, pt in model_torch.named_parameters():
        param_groups_torch.append(nt)
        assert pt.grad is not None

        numweights_torch.append(torch.numel(pt))

    assert numweights_model == numweights_torch
    assert len(param_groups_model) == 10
    assert len(param_groups_torch) == 10
    assert x_batch.size() == torch.Size([100, 10, 1])
    assert torch.allclose(train_loss, train_loss_torch)
    assert torch.allclose(my_test_loss, torch_test_loss)
