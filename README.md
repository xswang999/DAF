## Method Overview

1. Train **DAFNet** (ResNet backbone + classifier).
2. Load **pseudo-label.csv**.
3. Align distributions with **C-BHCC** and **D-BHCC**.

## Repository Structure

```
├── config/                 # EasyDict defaults
│   └── config.py
├── data/                   # Datasets & dataloaders
│   ├── categorical_dataset.py
│   ├── class_aware_dataset_dataloader.py
│   ├── custom_dataset_dataloader.py
│   ├── image_folder.py
│   ├── pairwise_dg_dataloader.py
│   ├── pseudo_label_loader.py
│   ├── single_dataset.py
│   └── utils.py
├── discrepancy/            # BHCC losses
│   ├── c_bhcc.py           # C-BHCC
│   └── d_bhcc.py           # D-BHCC
├── model/                  # Network definitions
│   ├── model.py            # DAFNet / daf_net
│   ├── resnet.py
│   ├── domain_specific_module.py
│   └── utils.py
├── solver/                 # Training utilities
│   ├── base_solver.py      # Optimizer, eval, checkpoint
│   ├── clustering.py       # Feature clustering
│   └── utils.py
├── tools/
│   ├── train.py            # Training entry point
│   └── prepare_data.py     # prepare_data_DAF()
└── utils/
    └── utils.py            # CUDA / accuracy
```

## Requirements

- Python 3.8
- PyTorch & torchvision
- numpy, Pillow, pandas
- PyYAML, easydict
- scipy
- CUDA

## Data Layout

```
experiments/dataset/Office-Home/
├── category.txt              # one class name per line (65 for Office-Home)
├── Art/<classname>/...
├── Clipart/<classname>/...
├── Product/<classname>/...
└── RealWorld/<classname>/...
```

```
results/Office-Home/{Labelled}_to_{Target}/
  step*_{Labelled}_to_{Unlabelled}/
    pseudo_labels_filtered.csv
```

CSV columns required: `image_path`, `pseudo_label`, `ground_truth`.