# focal-loss

PyTorch implementation of Focal Loss as described in the [original paper](https://arxiv.org/abs/1708.02002v2). Should be used as a replacement for `torch.nn.CrossEntropyLoss` when one wants to prioritise the resolution of difficult misclassification. 

## Usage

```
import torch

# Define batch size
BATCH = 16

# Define number of classes (multi-class case is possible)
CLASSES = 2

# Define the gamma parameter, the higher the more violent the dampening of 
# over-confident predictions
GAMMA = 2

# Define the weights for each class in case of imbalanced dataset.
# If not given, or all set to one, then no different weight is given
weight = torch.ones(CLASSES)

# Construct logits (softmax will be applied internally)
logits = torch.randn((BATCH, CLASSES))

# Construct targets (integers representing class indices)
targets = torch.randint(0, CLASSES, (BATCH, ))

# Instantiate object
criterion = FocalLoss(gamma = GAMMA, weight = WEIGHT)

# Apply loss function
loss = criterion(logits, targets)

```
