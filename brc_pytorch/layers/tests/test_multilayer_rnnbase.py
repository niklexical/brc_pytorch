"""Test Multilayer implementation by comparing it with in-built functions"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brc_pytorch.datasets import BRCDataset
from brc_pytorch.layers import MultiLayerBase, SelectItem


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


def test_rnn_dimensions(generate_dataset):
    input_size = 2
    hidden_size = 10
    batch_first = [True, False]
    bidirectional = [True, False]
    num_layers = [1, 2, 10]
    test_x = torch.randn(16, 5, 2)

    for batch in batch_first:
        if batch is False:
            test_x = test_x.permute(1, 0, 2)
        for bi in bidirectional:
            num_directions = 2 if bi else 1
            for layers in num_layers:

                recurrent_layers = [nn.GRUCell(input_size, hidden_size)]

                inner_input_dimensions = num_directions * hidden_size

                for _ in range(layers - 1):
                    recurrent_layers.append(
                        nn.GRUCell(inner_input_dimensions, hidden_size)
                    )

                assert len(recurrent_layers) == layers

                rnn = MultiLayerBase(
                    "GRU",
                    recurrent_layers,
                    hidden_size,
                    batch_first=batch,
                    bidirectional=bi,
                    return_sequences=True,
                )

                rnn_torch = nn.GRU(
                    input_size,
                    hidden_size,
                    num_layers=layers,
                    batch_first=batch,
                    bidirectional=bi
                )

                output_rnn = rnn(test_x)
                output_torch = rnn_torch(test_x)

                assert isinstance(output_rnn, tuple)

                assert len(output_rnn) == len(output_torch)

                for i in range(len(output_rnn)):
                    assert output_rnn[i].size() == output_torch[i].size()


def test_lstm_dimensions(generate_dataset):
    input_size = 2
    hidden_size = 10
    batch_first = [True, False]
    bidirectional = [True, False]
    num_layers = [1, 2, 10]
    test_x = torch.randn(16, 5, 2)

    for batch in batch_first:
        if batch is False:
            test_x = test_x.permute(1, 0, 2)
        for bi in bidirectional:
            num_directions = 2 if bi else 1
            for layers in num_layers:

                recurrent_layers = [nn.LSTMCell(input_size, hidden_size)]

                inner_input_dimensions = num_directions * hidden_size

                for _ in range(layers - 1):
                    recurrent_layers.append(
                        nn.LSTMCell(inner_input_dimensions, hidden_size)
                    )

                assert len(recurrent_layers) == layers

                rnn = MultiLayerBase(
                    "LSTM",
                    recurrent_layers,
                    hidden_size,
                    batch_first=batch,
                    bidirectional=bi,
                    return_sequences=True,
                )

                rnn_torch = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers=layers,
                    batch_first=batch,
                    bidirectional=bi
                )

                outputs_rnn = rnn(test_x)
                outputs_torch = rnn_torch(test_x)

                assert isinstance(outputs_rnn, tuple)
                assert len(outputs_rnn) == len(outputs_torch)

                out_rnn, (h_rnn, c_rnn) = outputs_rnn
                out_torch, (h_torch, c_torch) = outputs_torch

                assert out_rnn.size() == out_torch.size()

                assert h_rnn.size() == h_torch.size()
                assert c_rnn.size() == c_torch.size()


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_multilayer_rnn(
    generate_dataset, batch_first, bidirectional, num_layers
):
    """Tests multilayer functionality for cells that output hidden state only."""

    input_size = 1
    hidden_size = 10

    num_directions = 2 if bidirectional else 1

    recurrent_layers = [nn.GRUCell(input_size, hidden_size)]

    inner_input_dimensions = num_directions * hidden_size

    for _ in range(num_layers - 1):
        recurrent_layers.append(
            nn.GRUCell(inner_input_dimensions, hidden_size)
        )

    assert len(recurrent_layers) == num_layers

    rnn = MultiLayerBase(
        "GRU",
        recurrent_layers,
        hidden_size,
        batch_first=batch_first,
        bidirectional=bidirectional,
        return_sequences=True,
    )

    model = nn.Sequential(
        rnn, SelectItem(0, -1, batch_first),
        nn.Linear(num_directions * hidden_size, 1)
    )

    model_torch = nn.Sequential(
        nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional
        ),
        SelectItem(0, -1, batch_first),
        nn.Linear(num_directions * hidden_size, 1),
    )

    loss_fn_model = nn.MSELoss()
    loss_fn_torch = nn.MSELoss()
    optimiser_model = torch.optim.Adam(model.parameters())
    optimiser_torch = torch.optim.Adam(model_torch.parameters())

    epochs = 30

    training_loader, inputs_test, outputs_test = generate_dataset
    test_x = torch.from_numpy(inputs_test)
    if batch_first is False:
        # data always has batch first, so permute to make len first
        test_x = test_x.permute(1, 0, 2)
    test_y = torch.from_numpy(outputs_test)

    for e in range(epochs):

        model.train()
        model_torch.train()

        for idx, (x_batch, y_batch) in enumerate(training_loader):

            if batch_first is False:
                # data always has batch first, so permute to make len first
                x_batch = x_batch.permute(1, 0, 2)

            pred_train = model(x_batch)
            pred_train_torch = model_torch(x_batch)

            assert pred_train.size() == pred_train_torch.size()

            if batch_first is False:

                # permute to make batch first before loss calc
                pred_train = pred_train.permute(1, 0, 2)
                pred_train_torch = pred_train_torch.permute(1, 0, 2)

            train_loss = loss_fn_model(pred_train.squeeze(1), y_batch)
            train_loss_torch = loss_fn_torch(
                pred_train_torch.squeeze(1), y_batch
            )

            optimiser_model.zero_grad()
            train_loss.backward()
            optimiser_model.step()

            optimiser_torch.zero_grad()
            train_loss_torch.backward()
            optimiser_torch.step()

    if batch_first:
        assert test_x.size() == torch.Size([100, 10, 1])
    else:
        assert test_x.size() == torch.Size([10, 100, 1])

    model.eval()
    my_test_pred = model(test_x)

    model_torch.eval()
    torch_test_pred = model_torch(test_x)

    if batch_first is False:
        my_test_pred = my_test_pred.permute(1, 0, 2)
        torch_test_pred = torch_test_pred.permute(1, 0, 2)

    my_test_loss = loss_fn_model(my_test_pred.squeeze(1), test_y)

    torch_test_loss = loss_fn_torch(torch_test_pred.squeeze(1), test_y)

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
    assert len(param_groups_model) == num_layers * 4 * num_directions + 2
    assert len(param_groups_torch) == len(param_groups_model)

    assert torch.allclose(train_loss, train_loss_torch)
    assert torch.allclose(my_test_loss, torch_test_loss)


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_multilayer_lstm(
    generate_dataset, batch_first, bidirectional, num_layers
):
    """Tests multilayer functionality for cells that output hidden and cell state."""

    input_size = 1
    hidden_size = 10

    num_directions = 2 if bidirectional else 1

    recurrent_layers = [nn.LSTMCell(input_size, hidden_size)]

    inner_input_dimensions = num_directions * hidden_size

    for _ in range(num_layers - 1):
        recurrent_layers.append(
            nn.LSTMCell(inner_input_dimensions, hidden_size)
        )

    rnn = MultiLayerBase(
        "LSTM",
        recurrent_layers,
        hidden_size,
        batch_first=batch_first,
        bidirectional=bidirectional,
        return_sequences=True,
    )

    model = nn.Sequential(
        rnn, SelectItem(0, -1, batch_first),
        nn.Linear(num_directions * hidden_size, 1)
    )

    model_torch = nn.Sequential(
        nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional
        ),
        SelectItem(0, -1, batch_first),
        nn.Linear(num_directions * hidden_size, 1),
    )

    loss_fn_model = nn.MSELoss()
    loss_fn_torch = nn.MSELoss()
    optimiser_model = torch.optim.Adam(model.parameters())
    optimiser_torch = torch.optim.Adam(model_torch.parameters())

    epochs = 30

    training_loader, inputs_test, outputs_test = generate_dataset
    test_x = torch.from_numpy(inputs_test)
    if batch_first is False:
        # data always has batch first, so permute to make len first
        test_x = test_x.permute(1, 0, 2)
    test_y = torch.from_numpy(outputs_test)

    for e in range(epochs):

        model.train()
        model_torch.train()

        for idx, (x_batch, y_batch) in enumerate(training_loader):

            if batch_first is False:
                # data comes out with batch first, so permute to make len first
                x_batch = x_batch.permute(1, 0, 2)

            pred_train = model(x_batch)
            pred_train_torch = model_torch(x_batch)

            if batch_first is False:
                pred_train = pred_train.permute(1, 0, 2)
                pred_train_torch = pred_train_torch.permute(1, 0, 2)

            train_loss = loss_fn_model(pred_train.squeeze(1), y_batch)
            train_loss_torch = loss_fn_torch(
                pred_train_torch.squeeze(1), y_batch
            )

            optimiser_model.zero_grad()
            train_loss.backward()
            optimiser_model.step()

            optimiser_torch.zero_grad()
            train_loss_torch.backward()
            optimiser_torch.step()

    model.eval()
    my_test_pred = model(test_x)

    model_torch.eval()
    torch_test_pred = model_torch(test_x)

    if batch_first is False:
        my_test_pred = my_test_pred.permute(1, 0, 2)
        torch_test_pred = torch_test_pred.permute(1, 0, 2)

    my_test_loss = loss_fn_model(my_test_pred.squeeze(1), test_y)

    torch_test_loss = loss_fn_torch(torch_test_pred.squeeze(1), test_y)

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
    assert len(param_groups_model) == (num_layers * 4 * num_directions + 2)
    assert len(param_groups_torch) == len(param_groups_model)

    assert torch.allclose(train_loss, train_loss_torch)
    assert torch.allclose(my_test_loss, torch_test_loss)
