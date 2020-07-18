[![Build
Status](https://travis-ci.com/niklexical/brc_pytorch.svg?branch=master)](https://travis-ci.com/niklexical/brc_pytorch)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# brc_pytorch
Pytorch implementation of bistable recurrent cell with baseline comparisons.

This repository contains the Pytorch implementation of the paper ["A bio-inspired bistable recurrent cell allows for long-lasting memory".](https://arxiv.org/abs/2006.05252)

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

input_size = 32
hidden_size = 16 

# Behaves like a nn.GRUCell
brc = BistableRecurrentCell(input_size, hidden_size)
nbrc = NeuromodulatedBistableRecurrentCell(input_size, hidden_size)

# Create a 3-layer nBRC (behaves like a nn.GRU)
sizes = [input_size, 16, 16]
nbrc_cells = [NeuromodulatedBistableRecurrentCell(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
three_layer_nbrc = MultiLayerBase('nBRC', nbrc_cells, sizes[1:])
```




## Validation studies

First, the implementations of both the BRC and nBRC are validated on the
Copy-First-Input task (Benchmark 1 from the original paper). Moreover, it is well known
that standard RNNs have difficulties in *discrete counting*, especially for
longer sequences (see
[NeurIPS 2015 paper](http://papers.nips.cc/paper/5857-inferring-algorithmic-patterns-with-stack-augmented-recurrent-nets)).
Secondly, we here identify the task of **Binary Addiiton** as another
task for which the nBRC is superior to LSTM & GRU which begs implications for a
set of tasks involving more explicit memorization. For both tasks, the
performances of BRC and nBRC are compared with that of the LSTM and GRU cells. 

### Copy-First-Input

The goal of this task is to correctly predict the number at the start of a sequence of a certain length. 

This task is reproduced from the paper - 2 layer model with 100 units each, trained on datasets with increasing sequence lengths - 5, 100, 300. The plot is obtained by taking a moving average of the training loss per gradient iteration with window size = 100 for lengths 100 and 300, and window size 20 for length 5. 

The results from Copy-First-Input task show trends similar to that in the paper, thus confirming their findings. It should, however, be noted that the absolute losses are higher than reported in the paper. This is mostly due to the training and testing sizes being much smaller, and no hyperparameter tuning being done. 

![copy-first-input](https://github.com/niklexical/brc_pytorch/raw/master/results/copy-first-input.png)

To reproduce this task do:
```py

```

### Binary Addition

Additional testing on Binary Addition was done to test the capabilities of these cells. The goal of this task is to correctly predict the sum of two binary numbers (in integer form).

Both single layer and 2 layer models, with constant hidden units 100, are evaluated based on the accuracy of their predictions.

The results from this task prove the usefulness of both the nBRC and BRC layers which consistently perform better than both the LSTM and GRU. Moreover, it is interesting to note the potential of nBRC in the binary addition task which is consistent around near perfect accuracy upto sequence length 60. The plots are obtained by averaging the results over 5 runs of the experiment and highlighting the standard error of the average.

![copy-first-input](https://github.com/niklexical/brc_pytorch/raw/master/results/binary_addition_1layer.png)

![copy-first-input](https://github.com/niklexical/brc_pytorch/raw/master/results/binary_addition_2layer.png)

While the Copy-First-Input task highlights the performance superiority of these cells over the conventional LSTM and GRU, the Binary Addition task, which requires counting, is witness to their usefulness beyond just long-lasting memory.

To reproduce this task do:
```py

```