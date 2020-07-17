import os
import sys
import argparse
import logging
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from brc_pytorch.utils.layers.neuromodulated_brc_layer import NeuromodulatedBistableRecurrentCellLayer
from brc_pytorch.utils.layers.brc_layer import BistableRecurrentCellLayer
from brc_pytorch.utils.layers.multilayer_rnnbase import MultiLayerBase
from brc_pytorch.datasets.BRCDataset import BRCDataset
from brc_pytorch.utils.layers.select_item import SelectItem


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
parser.add_argument(
    'model_path', type=str, help='Path to save the best performing model.'
)
parser.add_argument(
    'results_path', type=str, help='Path to save the loss plots.'
)

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(model_path: str, results_path: str) -> None:
    """Execute copy-first-input for each cell and sequence length specified."

    Args:
        model_path (string): Path where the best model should be saved.
        results_path (string): Path where the results should be saved.

    """

    logging.basicConfig(
        handlers=[
            logging.
            FileHandler(os.path.join(results_path, "BRC_benchmark1.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('Benchmark1_BRC')
    logger.setLevel(logging.DEBUG)

    zs = [5, 100, 300]
    test_size = 5000

    lines = ['dotted', '-', '--']
    colours = sns.color_palette("husl", 5)
    line = -1

    for z in zs:
        line += 1
        c = -1
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.set_title(
            f'Copy-First-Input Training Loss of Various Recurrent Cells for Sequence Length {z}',
            fontsize=12,
            fontweight='medium'
        )
        ax.set_xlabel('Number of Gradient Iterations')
        ax.set_ylabel('MSE Loss')

        for name, cell in zip(
            ["LSTM", "GRU", "nBRC", "BRC"], [
                nn.LSTMCell, nn.GRUCell,
                NeuromodulatedBistableRecurrentCellLayer,
                BistableRecurrentCellLayer
            ]
        ):
            c += 1
            save_here = os.path.join(model_path, f'{name}_{z}')
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

            inputs_train = np.expand_dims(inputs, axis=2).astype(np.float32
                                                                 )[:-test_size]

            inputs_test = np.expand_dims(inputs, axis=2).astype(np.float32
                                                                )[-test_size:]

            outputs_train = np.expand_dims(outputs,
                                           axis=1).astype(np.float32
                                                          )[:-test_size]

            outputs_test = np.expand_dims(outputs,
                                          axis=1).astype(np.float32
                                                         )[-test_size:]

            dataset_train = BRCDataset(inputs_train, outputs_train)
            dataset_test = BRCDataset(inputs_test, outputs_test)

            training_loader = DataLoader(
                dataset_train, batch_size=100, shuffle=True
            )
            test_loader = DataLoader(
                dataset_test, batch_size=100, shuffle=True
            )
            """ Create Layers and add to Sequential"""

            logger.info(
                "Training network with cells of type {} with a lag of {} time-steps"
                .format(name, z)
            )
            logger.info("---------------------")

            input_size = inputs_train.shape[2]
            hidden_sizes = [input_size, 100, 100]

            recurrent_layers = [
                cell(hidden_sizes[i], hidden_sizes[i + 1])
                for i in range(len(hidden_sizes) - 1)
            ]

            rnn = MultiLayerBase(
                name, recurrent_layers, hidden_sizes[1:], device
            )

            if name == "LSTM":
                model = nn.Sequential(
                    rnn, SelectItem(0), nn.Linear(hidden_sizes[2], 1)
                ).to(device)
            else:
                model = nn.Sequential(rnn, nn.Linear(hidden_sizes[2],
                                                     1)).to(device)
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

                        x_batch, y_batch = x_batch.to(device
                                                      ), y_batch.to(device)

                        pred_train = model(x_batch)
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
                        "Train Loss = {}".format(
                            train_loss.data.cpu().numpy()
                        )
                    )
                    train_epochs_avg_losses.append(train_loss_epoch_avg)

                    model.eval()
                    test_loss_epoch_avg = 0
                    for idx, (x_test, y_test) in enumerate(test_loader):

                        x_test, y_test = x_test.to(device), y_test.to(device)
                        pred_test = model(x_test)

                        test_loss = loss_fn(pred_test, y_test)
                        test_allepochs_losses.append(
                            test_loss.data.cpu().numpy()
                        )
                        test_loss_epoch_avg = (
                            test_loss_epoch_avg * idx +
                            test_loss.data.cpu().numpy()
                        ) / (idx + 1)

                    logger.info("Test Loss = {}".format(test_loss_epoch_avg))
                    test_allepochs_losses.append(test_loss_epoch_avg)

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
                os.path.join(results_path, f'TrainLoss_AllE_{name}_{z}'),
                train_allepochs_losses
            )
            np.save(
                os.path.join(results_path, f'TrainAvgLoss_AllE_{name}_{z}'),
                train_epochs_avg_losses
            )
            np.save(
                os.path.join(results_path, f'ValidLoss_AllE_{name}_{z}'),
                test_allepochs_losses
            )

            np.save(
                os.path.join(results_path, f'ValidAvgLoss_AllE_{name}_{z}'),
                test_epochs_avg_losses
            )

            ax.plot(
                range(grad_iterations),
                train_allepochs_losses,
                ls=lines[line],
                color=colours[c]
            )
            fig.savefig(
                os.path.join(
                    results_path, f'Training_{name}{z}_benchmark1.png'
                )
            )

        lgd = ["LSTM", "GRU", "nBRC", "BRC"]
        lgd1 = fig.legend(lgd, bbox_to_anchor=(1.04, 0.5), loc="center left")

        fig.savefig(
            os.path.join(results_path, f'Training_{z}_benchmark1.png'),
            bbox_extra_artists=(lgd1, ),
            bbox_inches='tight'
        )

        fig.clf()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.model_path, args.results_path)
