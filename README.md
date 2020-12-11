[![PyPI version](https://badge.fury.io/py/brc-pytorch.svg)](https://badge.fury.io/py/brc-pytorch)
[![Build
Status](https://travis-ci.com/niklexical/brc_pytorch.svg?branch=master)](https://travis-ci.com/niklexical/brc_pytorch)
[![codecov](https://codecov.io/gh/niklexical/brc_pytorch/branch/master/graph/badge.svg?token=UQ5O5CP8KD)](https://codecov.io/gh/niklexical/brc_pytorch)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# brc_pytorch
Pytorch implementation of bistable recurrent cell with baseline comparisons.

This repository contains the Pytorch implementation of the paper ["A bio-inspired bistable recurrent cell allows for long-lasting memory"](https://arxiv.org/abs/2006.05252). The original `tensorflow` implementation by the author Nicolas Vecoven can be found [here](https://github.com/nvecoven/BRC).

Another important feature of this repository is the implementation of a base class that returns a recurrent neural network for a given recurrent cell. Based on the hyperparameters provided, the network can have multiple layers, be bidirectional and the input can either have batch first or not. The outputs from the network mimic that returned by GRU/LSTM networks developed by PyTorch, with an additional option of returning only the hidden states from the last layer and last time step.
## Package setup

`brc_pytorch` is `pypi` installable:
```sh
pip install brc_pytorch
```
### Development setup
Create a `venv`, activate it, install dependencies and package in editable mode.
```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Usage (example)
```py
from brc_pytorch.layers import BistableRecurrentCell, NeuromodulatedBistableRecurrentCell
from brc_pytorch.layers import MultiLayerBase

# Create a 3-layer nBRC (behaves like a nn.GRU)

input_size = 32
hidden_size = 16
num_layers = 3
bidirectional = True
batch_first = True
return_sequences = False

num_directions = 2 if bidirectional else 1

# Behaves like a nn.GRUCell
nbrc = NeuromodulatedBistableRecurrentCell(input_size, hidden_size)

# Append cells for subsequent layers keeping in mind
for _ in range(num_layers - 1):
        nbrc.append(
            NeuromodulatedBistableRecurrentCell(inner_input_dimensions, hidden_size)
        )

three_layer_nbrc = rnn = MultiLayerBase(
        "nBRC",
        nbrc,
        hidden_size,
        batch_first,
        bidirectional,
        return_sequences,
    )
```


## Validation studies

First, the implementations of both the BRC and nBRC are validated on the
Copy-First-Input task (Benchmark 1 from the original paper). Moreover, it is well known
that standard RNNs have difficulties in *discrete counting*, especially for
longer sequences (see
[NeurIPS 2015 paper](http://papers.nips.cc/paper/5857-inferring-algorithmic-patterns-with-stack-augmented-recurrent-nets)).
To this end, we here identify **Binary Addition** as another
task for which the nBRC is superior to LSTM & GRU which begs implications for a
set of tasks involving more explicit memorization. For both tasks, the
performances of BRC and nBRC are compared with that of the LSTM and GRU cells. 

### Copy-First-Input

The goal of this task is to correctly predict the number at the start of a sequence of a certain length. 

This task is reproduced from the paper - 2 layer model with 100 units each, trained on datasets with increasing sequence lengths - 5, 100, 300. The plot is obtained by taking a moving average of the training loss per gradient iteration with window size = 100 for lengths 100 and 300, and window size 20 for length 5. 

The results from Copy-First-Input task show trends similar to that in the paper, thus confirming their findings. It should, however, be noted that the absolute losses are higher than reported in the paper. This is mostly due to the training and testing sizes being much smaller, and no hyperparameter tuning being done. 

![copy-first-input](https://github.com/niklexical/brc_pytorch/raw/master/results/copy-first-input.png)

To reproduce this task do:
1. Change directory to the `scripts` folder. From the terminal, run the following commands:
```sh
# The following command creates a directory with subdirectories in the scripts folder to save the models and results.
mkdir -p test_benchmark1/{models,results}
# Run the training script with your python executable. The following is an example for Anaconda.
/opt/anaconda3/envs/venv/bin/python brc_benchmark1.py test_benchmark1/models/ test_benchmark1/results/

```
Or, if training takes a very long time, run the script cell-wise, i.e, specify cell name as an additional argument and run multiple jobs in parallell - one for each cell.
```sh
/opt/anaconda3/envs/venv/bin/python brc_benchmark1_cell.py nBRC test_benchmark1/models/ test_benchmark1/results/

```
2. Calculate the moving average for each `TrainLoss_AllE_*.npy` file from test_benchmark1/results/ and plot.

### Binary Addition

Additional testing on Binary Addition was done to test the capabilities of these cells. The goal of this task is to correctly predict the sum of two binary numbers (in integer form).

Both single layer and 2 layer models, with constant hidden units 100, are evaluated based on the accuracy of their predictions.

The results from this task prove the usefulness of both the nBRC and BRC layers which consistently perform better than both the LSTM and GRU. Moreover, it is interesting to note the potential of nBRC in the binary addition task which is consistent around near perfect accuracy upto sequence length 60. The plots are obtained by averaging the results over 5 runs of the experiment and highlighting the standard error of the average.

![copy-first-input](https://github.com/niklexical/brc_pytorch/raw/master/results/binary_addition_1layer.png)

![copy-first-input](https://github.com/niklexical/brc_pytorch/raw/master/results/binary_addition_2layer.png)

While the Copy-First-Input task highlights the performance superiority of these cells over the conventional LSTM and GRU, the Binary Addition task, which requires counting, is witness to their usefulness beyond just long-lasting memory.

To reproduce this task do:

1. Change directory to the `scripts` folder. From the terminal, run the following command:
```sh
# The following command creates a directory with subdirectories in the scripts folder to save the models and results.
mkdir -p test_binary_addition/{models,results}/{test1,test2,test3,test4,test5}

```
2. Create and run the following python script from the same directory. Make sure the python executable file is correct.
```py
import os
import sys
import subprocess

dir_models = 'test_binary_addition/models/'
dir_results = 'test_binary_addition/results/'

modelpaths = [
    os.path.join(dir_models,f'test{i}') for i in range(1,6)
]
resultpaths = [
    os.path.join(dir_results,f'test{i}') for i in range(1,6)
]

procs = []
for i in range(5):
    proc = subprocess.Popen(
        [
            sys.executable,
            'binary_addition_train.py',
            modelpaths[i], resultpaths[i]
        ]
    )
    procs.append(proc)

for proc in procs:
    proc.wait()
```

3. Calculate the mean and standard error of mean over the different tests for each `test_acc_*.npy` file and plot.

For the 2 layer implementation, simply add another 100 to the `hidden_sizes` variable in the training file, and repeat the steps.
