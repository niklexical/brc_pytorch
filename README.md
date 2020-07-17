# brc_pytorch
Pytorch implementation of bistable recurrent cell with baseline comparisons.

This repository contains the Pytorch implementation of the paper ["A bio-inspired bistable recurrent cell allows for long-lasting memory"] (https://arxiv.org/abs/2006.05252).

The implementations of both the bistable recurrent cell and its neurmodulated version are tested on 2 tasks - Copy-First-Input (Benchmark 1 from https://arxiv.org/abs/2006.05252) and Binary Addition. Their performances are compared with that of the LSTM and GRU cells. 

## Copy-First-Input

The goal of this task is to correctly predict the number at the start of a sequence of a certain length. 

This task is reproduced from the paper - 2 layer model with 100 units each, trained on datasets with increasing sequence lengths - 5, 100, 300. 

The results from Copy-First-Input task show trends similar to that in the paper, thus confirming their findings. It should, however, be noted that the absolute losses are higher than reported in the paper. This is mostly due to the training and testing sizes being much smaller, and no hyperparameter tuning being done. 

![copy-first-input](https://github.com/niklexical/brc_pytorch/raw/master/results/copy-first-input.png)

## Binary Addition

Additional testing on Binary Addition was done to test the capabilities of these cells. The goal of this task is to correctly predict the sum of two binary numbers (in integer form).

Both single layer and 2 layer models, with constant hidden units 100, are evaluated based on the accuracy of their predictions.

The results from this task prove the usefulness of both the nBRC and BRC layers which consistently perform better than both the LSTM and GRU. Moreover, it is interesting to note the potential of nBRC in the binary addition task which is consistent around near perfect accuracy upto sequence length 60. 

![copy-first-input](https://github.com/niklexical/brc_pytorch/raw/master/results/binary_addition_1layer.png)

![copy-first-input](https://github.com/niklexical/brc_pytorch/raw/master/results/binary_addition_2layer.png)

While the Copy-First-Input task highlights the performance superiority of these cells over the conventional LSTM and GRU, the Binary Addition task, which requires counting, is witness to their usefulness beyond just long-lasting memory.
