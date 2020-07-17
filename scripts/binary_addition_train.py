import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brc_pytorch.datasets import BinaryAddition
from brc_pytorch.layers import (
    BistableRecurrentCell, MultiLayerBase, NeuromodulatedBistableRecurrentCell,
    SelectItem
)

parser = argparse.ArgumentParser()
parser.add_argument(
    'model_path', type=str, help='Path to save the best performing model.'
)
parser.add_argument(
    'results_path',
    type=str,
    help='Path to save the training and validation losses.'
)

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(model_path: str, results_path: str) -> None:
    """Executes binary addition task for each cell sequentially.

    Args:
        model_path (string): Path where the best model should be saved.
        results_path (string): Path where the results should be saved.

    """

    # setup logging
    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join(results_path, "Binary_Addition.log")
            ),
            logging.StreamHandler(sys.stdout)
        ],
    )
    logger = logging.getLogger('BinAdd_BRC')
    logger.setLevel(logging.DEBUG)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title(
        'Train and Validation Loss of Various Recurrent Cells on Binary Addition'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

    fig2, ax2 = plt.subplots(constrained_layout=True)
    ax2.set_title(
        'Test Accuracy of Various Recurrent Cells on Sequences upto Length 60'
    )
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Accuracy")

    fig3, ax3 = plt.subplots(constrained_layout=True)
    ax3.set_title(
        'Train and Validation Accuracy of Various Recurrent Cells on Binary Addition'
    )
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Accuracy")

    colours = sns.color_palette("Paired", 8)

    c = 0
    for name, cell in zip(
        ["LSTM", "GRU", "nBRC", "BRC"], [
            nn.LSTMCell, nn.GRUCell, NeuromodulatedBistableRecurrentCell,
            BistableRecurrentCell
        ]
    ):
        save_here = os.path.join(model_path, f'{name}_binadd')
        model = None
        if model is not None:
            del (model)
        """Create Train and Test Dataset"""

        sample_size_train = 10000
        sample_size_valid = 1000
        sample_size_test = 1000
        max_len_train = 20
        min_len_train = 3
        max_len_test = 60
        epochs = 15
        min_len_test = 3

        plot_train_loss = []
        plot_valid_loss = []
        plot_test_loss = []
        plot_test_accuracy = []
        plot_accuracy_train = []
        plot_accuracy_valid = []

        input_size = 2
        hidden_sizes = [input_size, 100]

        recurrent_layers = [
            cell(hidden_sizes[i], hidden_sizes[i + 1])
            for i in range(len(hidden_sizes) - 1)
        ]

        rnn = MultiLayerBase(
            name,
            recurrent_layers,
            hidden_sizes[1:],
            device,
            return_sequences=True
        )

        model = nn.Sequential(
            rnn, SelectItem(0), nn.Linear(hidden_sizes[-1], 1), nn.Sigmoid()
        ).to(device)

        loss_fn = nn.BCELoss()
        optimiser = torch.optim.Adam(model.parameters())

        min_loss = np.inf

        for i in range(min_len_train, max_len_train + 1):

            sequence_length = i

            dataset_train = BinaryAddition(
                sample_size_train, sequence_length, max_len_train,
                min_len_train, 'single'
            )
            dataset_valid = BinaryAddition(
                sample_size_valid, sequence_length, max_len_train,
                min_len_train, 'single'
            )

            training_loader = DataLoader(
                dataset_train, batch_size=200, shuffle=True
            )
            valid_loader = DataLoader(
                dataset_valid, batch_size=200, shuffle=True
            )

            logger.info(
                "Training network with cells of type {}, len {}".format(
                    name, i
                )
            ),
            logger.info("---------------------")

            with torch.autograd.set_detect_anomaly(True):

                for epoch in range(epochs):

                    model.train()
                    logger.info("=== Epoch [{}/{}]".format(epoch + 1, epochs))

                    for idx, (x_batch, y_batch) in enumerate(training_loader):

                        x_batch, y_batch = x_batch.to(device
                                                      ), y_batch.to(device)

                        pred_train = model(x_batch)

                        train_loss = loss_fn(pred_train, y_batch)
                        optimiser.zero_grad()
                        train_loss.backward()
                        optimiser.step()

                    pred_seq_train = torch.round(pred_train)

                    correct_train = (pred_seq_train == y_batch).all(
                        dim=1
                    ).float().sum()

                    accuracy_train = correct_train / len(y_batch)

                    plot_accuracy_train.append(accuracy_train)

                    plot_train_loss.append(train_loss.data.cpu().numpy())
                    logger.info(
                        "Train Loss = {}".format(
                            train_loss.data.cpu().numpy()
                        )
                    )

                    model.eval()
                    for x_valid, y_valid in valid_loader:

                        x_valid, y_valid = x_valid.to(device
                                                      ), y_valid.to(device)
                        pred_valid = model(x_valid)

                        valid_loss = loss_fn(pred_valid, y_valid)

                    pred_seq_valid = torch.round(pred_valid)
                    correct_valid = (pred_seq_valid == y_valid).all(
                        dim=1
                    ).float().sum()
                    accuracy_valid = correct_valid / len(y_valid)

                    plot_accuracy_valid.append(accuracy_valid)

                    plot_valid_loss.append(valid_loss.data.cpu().numpy())
                    logger.info(
                        "Validation Loss = {}".format(
                            valid_loss.data.cpu().numpy()
                        )
                    )

                    if valid_loss < min_loss:
                        min_loss = valid_loss
                        torch.save(
                            {
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimiser.state_dict(),
                                'train_loss': train_loss,
                                'test_loss': valid_loss,
                                'sequence_length': i
                            }, save_here
                        )

        np.save(
            os.path.join(results_path, f'train_loss_{name}'), plot_train_loss
        )
        np.save(
            os.path.join(results_path, f'train_acc_{name}'),
            plot_accuracy_train
        )
        np.save(
            os.path.join(results_path, f'valid_loss_{name}'), plot_valid_loss
        )
        np.save(
            os.path.join(results_path, f'valid_acc_{name}'),
            plot_accuracy_valid
        )

        ax.plot(
            range(epochs * (max_len_train + 1 - min_len_train)),
            plot_train_loss,
            color=colours[c + 1]
        )
        ax.plot(
            range(epochs * (max_len_train + 1 - min_len_train)),
            plot_valid_loss,
            color=colours[c]
        )

        ax3.plot(
            range(epochs * (max_len_train + 1 - min_len_train)),
            plot_accuracy_train,
            color=colours[c + 1]
        )
        ax3.plot(
            range(epochs * (max_len_train + 1 - min_len_train)),
            plot_accuracy_valid,
            color=colours[c]
        )

        for i in range(3, max_len_test + 1):

            sequence_length = i

            dataset_test = BinaryAddition(
                sample_size_test, sequence_length, max_len_test, min_len_test,
                'single'
            )

            test_loader = DataLoader(
                dataset_test, batch_size=200, shuffle=True
            )
            model.eval()

            test_losses = []
            accuracy_test = []

            for idx, (x_test, y_test) in enumerate(test_loader):

                x_test, y_test = x_test.to(device), y_test.to(device)
                pred_test = model(x_test)
                pred_seq_test = torch.round(pred_test)

                test_loss = loss_fn(pred_test, y_test)

                correct_test = (pred_seq_test == y_test).all(dim=1
                                                             ).float().sum()

                accuracy_test.append(correct_test / len(y_test))

                test_losses.append(test_loss.data.cpu().numpy())

            avg_test_loss = np.mean(test_losses)
            avg_accuracy = torch.mean(torch.as_tensor(accuracy_test))

            logger.info("Avg Test Loss = {}".format(avg_test_loss))
            logger.info("Avg Test Accuracy = {}".format(avg_accuracy))

            plot_test_loss.append(avg_test_loss)
            plot_test_accuracy.append(avg_accuracy)

        np.save(
            os.path.join(results_path, f'test_loss_{name}'), plot_test_loss
        )

        np.save(
            os.path.join(results_path, f'test_acc_{name}'), plot_test_accuracy
        )

        ax2.plot(
            range(3, max_len_test + 1),
            plot_test_accuracy,
            color=colours[c + 1]
        )

        c += 2
        np.save(
            os.path.join(results_path, f'test_loss_{name}'), plot_test_loss
        )

    legends_loss = sum(
        [
            [f'Training Loss {cell}', f'Validation Loss {cell}']
            for cell in ["LSTM", "GRU", "nBRC", "BRC"]
        ], []
    )
    legends_acc = sum(
        [
            [f'Training Acc {cell}', f'Validation Acc {cell}']
            for cell in ["LSTM", "GRU", "nBRC", "BRC"]
        ], []
    )
    lgd1 = fig.legend(
        legends_loss, bbox_to_anchor=(1.04, 0.5), loc="center left"
    )
    lgd2 = fig2.legend(
        ["LSTM", "GRU", "nBRC", "BRC"],
        bbox_to_anchor=(1.04, 0.5),
        loc="center left"
    )
    lgd3 = fig3.legend(
        legends_acc, bbox_to_anchor=(1.04, 0.5), loc="center left"
    )

    fig.savefig(
        os.path.join(results_path, 'TVLoss_binary_addition.png'),
        bbox_extra_artists=(lgd1, ),
        bbox_to_anchor=(1.04, 1),
        bbox_inches='tight'
    )

    fig2.savefig(
        os.path.join(results_path, 'TestAcc_binary_addition.png'),
        bbox_extra_artists=(lgd2, ),
        bbox_to_anchor=(1.04, 1),
        bbox_inches='tight'
    )

    fig3.savefig(
        os.path.join(results_path, 'TVAcc_binary_addition.png'),
        bbox_extra_artists=(lgd3, ),
        bbox_to_anchor=(1.04, 1),
        bbox_inches='tight'
    )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.model_path, args.results_path)
