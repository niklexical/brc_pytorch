"""Cell wise implementation of copy-first-input task"""
import argparse
import logging
import os
import sys
from typing import Tuple
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brc_pytorch.datasets import BRCDataset
from brc_pytorch.layers.brc_layer import BistableRecurrentCell
from brc_pytorch.layers.multilayer_rnnbase import MultiLayerBase
from brc_pytorch.layers.neuromodulated_brc_layer import \
    NeuromodulatedBistableRecurrentCell


def permute_reshape(output, batch, num_directions, hidden_size):
    output = output.permute(1, 0, 2)
    assert output.size() == torch.Size([batch, num_directions, hidden_size])
    output = output.reshape(batch, num_directions * hidden_size)
    return output


def generate_sample(n: int) -> Tuple:
    """Generates 1D data.

    Args:
        n (int): Lag size.

    Returns:
        Tuple: Tuple of 1D time series and the true value n steps behind.
    """
    true_n = np.random.randn()
    chain = np.concatenate([[true_n], np.random.randn(n - 1)])
    return chain, true_n


parser = argparse.ArgumentParser()
parser.add_argument('cell_name', type=str, help='Recurrent cell to test.')
parser.add_argument(
    'model_path', type=str, help='Path to save the best performing model.'
)
parser.add_argument(
    'results_path', type=str, help='Path to save the loss plots.'
)

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(cell_name: str, model_path: str, results_path: str) -> None:
    """Executes copy-first-input task for specified cell.

    Args:
        cell_name (string): Name of the recurrent cell to be used. One of
            ["LSTM","GRU","nBRC","BRC"].
        model_path (string): Path where the best model should be saved.
        results_path (string): Path where the results should be saved.

    """

    # setup logging
    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join(results_path, f"BRC_benchmark1_{cell_name}.log")
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('Benchmark1_BRC')
    logger.setLevel(logging.DEBUG)

    zs = [5, 100, 300]
    test_size = 5000

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.set_title(
        f'Copy-First-Input Training Loss of {cell_name}',
        fontsize=12,
        fontweight='medium'
    )
    ax.set_xlabel('Number of Gradient Iterations')
    ax.set_ylabel('MSE Loss')

    lines = ['dotted', '-', '--']
    colours = sns.color_palette("husl", 5)

    cell_modules = [
        nn.LSTMCell, nn.GRUCell, NeuromodulatedBistableRecurrentCell,
        BistableRecurrentCell
    ]

    cell_idx = ["LSTM", "GRU", "nBRC", "BRC"].index(cell_name)

    cell = cell_modules[cell_idx]

    line = 0

    for z in zs:

        save_here = os.path.join(model_path, f'{cell_name}_{z}')

        model = None
        if model is not None:
            del (model)
        """Create Train and Test Dataset"""

        inputs = []
        outputs = []

        for i in range(50000):
            inp, out = generate_sample(z)
            inputs.append(inp)
            outputs.append(out)

        inputs = np.array(inputs)
        outputs = np.array(outputs)

        inputs_train = np.expand_dims(inputs,
                                      axis=2).astype(np.float32)[:-test_size]

        inputs_test = np.expand_dims(inputs,
                                     axis=2).astype(np.float32)[-test_size:]

        outputs_train = np.expand_dims(outputs,
                                       axis=1).astype(np.float32)[:-test_size]

        outputs_test = np.expand_dims(outputs,
                                      axis=1).astype(np.float32)[-test_size:]

        dataset_train = BRCDataset(inputs_train, outputs_train)
        dataset_test = BRCDataset(inputs_test, outputs_test)

        training_loader = DataLoader(
            dataset_train, batch_size=100, shuffle=True
        )
        test_loader = DataLoader(dataset_test, batch_size=100, shuffle=True)
        """ Create Layers and add to Sequential"""

        logger.info(
            "Training network with cells of type {} with a lag of {} time-steps"
            .format(cell_name, z)
        )
        logger.info("---------------------")

        input_size = inputs_train.shape[2]
        hidden_size = 100
        num_layers = 2
        bidirectional = False

        num_directions = 2 if bidirectional else 1
        inner_input_dimensions = num_directions * hidden_size

        recurrent_layers = [cell(input_size, hidden_size)]

        for _ in range(num_layers - 1):
            recurrent_layers.append(cell(inner_input_dimensions, hidden_size))

        rnn = MultiLayerBase(
            cell_name,
            recurrent_layers,
            hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            return_sequences=False,
            device=device
        )

        fc = nn.Linear(hidden_size * num_directions, 1)

        get_hidden = itemgetter(0)

        model = nn.ModuleDict({'rnn': rnn, 'fc': fc}).to(device)

        loss_fn = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters())

        epochs = 60
        min_loss = np.inf

        train_allepochs_losses = []
        test_allepochs_losses = []
        train_epochs_avg_losses = []
        test_epochs_avg_losses = []

        with torch.autograd.set_detect_anomaly(True):
            grad_iterations = 0
            for e in range(epochs):
                train_loss_epoch_avg = 0
                model.train()
                logger.info("=== Epoch [{}/{}]".format(e + 1, epochs))

                for idx, (x_batch, y_batch) in enumerate(training_loader):

                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                    pred_train = model['rnn'](x_batch)

                    if cell_name == "LSTM":
                        pred_train = get_hidden(pred_train)

                    pred_train = permute_reshape(
                        pred_train, x_batch.size(0), num_directions,
                        hidden_size
                    )

                    pred_train = model['fc'](pred_train)

                    train_loss = loss_fn(pred_train, y_batch)
                    optimiser.zero_grad()
                    train_loss.backward()
                    optimiser.step()
                    train_allepochs_losses.append(
                        train_loss.data.cpu().numpy()
                    )
                    grad_iterations += 1
                    train_loss_epoch_avg = (
                        train_loss_epoch_avg * idx +
                        train_loss.data.cpu().numpy()
                    ) / (idx + 1)

                logger.info(
                    "Train Loss = {}".format(train_loss.data.cpu().numpy())
                )
                train_epochs_avg_losses.append(train_loss_epoch_avg)

                model.eval()
                test_loss_epoch_avg = 0

                for idx, (x_test, y_test) in enumerate(test_loader):

                    x_test, y_test = x_test.to(device), y_test.to(device)
                    pred_test = model['rnn'](x_test)

                    if cell_name == "LSTM":
                        pred_test = get_hidden(pred_test)

                    pred_train = permute_reshape(
                        pred_test, x_test.size(0), num_directions, hidden_size
                    )

                    pred_test = model['fc'](pred_test)

                    test_loss = loss_fn(pred_test, y_test)
                    test_allepochs_losses.append(test_loss.data.cpu().numpy())
                    test_loss_epoch_avg = (
                        test_loss_epoch_avg * idx +
                        test_loss.data.cpu().numpy()
                    ) / (idx + 1)

                logger.info("Test Loss = {}".format(test_loss_epoch_avg))
                test_epochs_avg_losses.append(test_loss_epoch_avg)

                if test_loss_epoch_avg < min_loss:
                    min_loss = test_loss_epoch_avg
                    torch.save(
                        {
                            'epoch': e,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimiser.state_dict(),
                            'train_loss': train_loss,
                            'test_loss': test_loss
                        }, save_here
                    )

                if test_loss_epoch_avg < 0.1:
                    break

        np.save(
            os.path.join(results_path, f'TrainLoss_AllE_{cell_name}_{z}'),
            train_allepochs_losses
        )
        np.save(
            os.path.join(results_path, f'TrainAvgLoss_AllE_{cell_name}_{z}'),
            train_epochs_avg_losses
        )

        np.save(
            os.path.join(results_path, f'ValidLoss_AllE_{cell_name}_{z}'),
            test_allepochs_losses
        )

        np.save(
            os.path.join(results_path, f'ValidAvgLoss_AllE_{cell_name}_{z}'),
            test_epochs_avg_losses
        )

        ax.plot(
            range(grad_iterations),
            train_allepochs_losses,
            ls=lines[line],
            color=colours[cell_idx]
        )
        fig.savefig(
            os.path.join(
                results_path, f'Training_{cell_name}{z}_benchmark1.png'
            )
        )
        line += 1

    lgd = fig.legend(
        [f"Length {zs[0]}", f"Length {zs[1]}", f"Length {zs[2]}"],
        bbox_to_anchor=(1.04, 0.5),
        loc="center left"
    )

    fig.savefig(
        os.path.join(results_path, f'{cell_name}_benchmark1.png'),
        bbox_extra_artists=(lgd, ),
        bbox_inches='tight'
    )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.cell_name, args.model_path, args.results_path)
