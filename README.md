# yakupred_model

PyTorch model implementation for predicting yaku from Mahjong game records, using the winning player's discard tiles as input.

This repository currently contains a minimal set of components for training. It does not include a CLI, pretrained models, or the dataset itself.

## Structure

```text
.
├── data/
│   ├── preprocessing.py  # Extracts discard tiles and yaku from paifu JSON
│   ├── encoding.py       # Encodes tiles and yaku labels
│   └── dataset.py        # Converts data into a PyTorch dataset format
├── models/
│   └── cnn_model.py      # CNN model definition
├── training/
│   └── train.py          # Training loop
├── evaluation/
│   └── evaluate.py       # Evaluation logic. Not implemented yet
└── README.md
```

## Requirements

- Python 3
- NumPy
- PyTorch

Example:

```bash
pip install numpy torch
```

## Data Format

`data.preprocessing.extract_single_paifu_data()` reads the `log` field from a paifu JSON object and extracts the following information for each round:

- The winning player's discard tiles
- The yaku present in the winning hand

Expected tile IDs use formats such as `m1`, `p5`, `s9`, and `z1`. If a tile ID ends with `_` or `*`, preprocessing removes the trailing character.

## Basic Usage

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.preprocessing import extract_single_paifu_data
from data.encoding import create_tile_images, create_yaku_dict, create_multi_hot_labels
from data.dataset import plus_set
from models.cnn_model import build_models
from training.train import train

# paifu_json_data is expected to be a preloaded paifu JSON dictionary
all_discard_tiles, all_roles = extract_single_paifu_data(paifu_json_data)

tile_images = create_tile_images(all_discard_tiles)
yaku_dict = create_yaku_dict(all_roles)
labels = create_multi_hot_labels(all_roles, yaku_dict)

dataset = plus_set(tile_images, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_models(num_labels=len(yaku_dict)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.BCEWithLogitsLoss()

losses = train(
    n_epochs=10,
    model=model,
    dataloder=dataloader,
    optimizer=optimizer,
    loss_function=loss_function,
    device=device,
)
```

## Input Tensor

`create_tile_images()` converts discard tile lists into `4 x 37` arrays.

- Rows: counts for repeated copies of the same tile
- Columns: tile type indices

`plus_set()` converts these arrays to `torch.float32` and adds a channel dimension. The model generally receives tensors with shape `(batch, 4, 1, 37)`.

## Model

`models.cnn_model.build_models(num_labels)` returns a CNN with the following structure:

- Feature extraction with several stacked `Conv2d`, `BatchNorm2d`, and `ReLU` layers
- Aggregation with `AdaptiveAvgPool2d((1, 4))`
- Fully connected layers and `Dropout`
- Output dimension of `num_labels`

Because the model is intended to predict multiple yaku at the same time, `BCEWithLogitsLoss` is a natural choice for the loss function.

## Notes

- `evaluation/evaluate.py` is not implemented yet.
- The argument name in `training/train.py` is currently `dataloder`.
- Data loading, pretrained model saving, and evaluation metric calculation are not implemented yet.
