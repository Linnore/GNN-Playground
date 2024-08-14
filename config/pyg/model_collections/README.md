# Model Collections

This repository contains a collection of models, each with its own configuration settings and parameters. It has nine models, which are GraphSAGE-benchmark-trans, GAT-benchmark-trans, GAT-benchmark-in, GIN-benchmark-trans, GIN-PyG-trans, PNA-PyG-trans, PNA-benchmark-trans, GINe-in, and PNAe-in models. Below is a detailed description of each model, including the base model, sampling strategy, weight decay, and others.

## Model


### Model 1: [GraphSAGE-benchmark-trans]
- **base_model**: "GraphSAGE_PyG"
- **framework**: "transductive"
- **sampling_strategy**: "SAGE"
- **lr**: 5e-3
- **weight_decay**: 5e-4
- **num_layers**: 2
- **num_neighbors**: [25, 10]
- **hidden_channels**: 64
- **dropout**: 0
- **aggr**: "mean"

### Model 2: [GAT-benchmark-trans]
- **base_model**: "GAT_Custom"
- **framework**: "transductive"
- **sampling_strategy**: "None"
- **lr**: 5e-3
- **weight_decay**: 5e-4
- **hidden_channels_per_head**: 8
- **num_layers**: 2
- **heads**: 8
- **output_heads**: 1
- **dropout**: 0.6
- **jk**: None
- **v2**: False

### Model 3: [GAT-benchmark-in]
- **base_model**: "GAT_Custom"
- **framework**: "inductive"
- **sampling_strategy**: "GraphBatching"
- **batch_size**: 2
- **lr**: 5e-3
- **weight_decay**: 0
- **hidden_channels_per_head**: [256, 256]
- **num_layers**: 3
- **heads**: [4, 4]
- **output_heads**: 1
- **dropout**: 0
- **jk**: None
- **v2**: False
- **skip_connection**: True

### Model 4: [GIN-benchmark-trans]
- **base_model**: "GIN_Custom"
- **framework**: "transductive"
- **sampling_strategy**: "SAGE"
- **lr**: 1e-3
- **weight_decay**: 1e-3
- **sample_when_predict**: True
- **hidden_channels**: 64
- **num_layers**: 2
- **num_neighbors**: [25, 10]
- **dropout**: 0
- **jk**: None
- **GINE**: False
- **skip_connection**: True

### Model 5: [GIN-PyG-trans]
- **base_model**: "GIN_PyG"
- **framework**: "transductive"
- **sampling_strategy**: "None"
- **lr**: 1e-2
- **weight_decay**: 5e-4
- **sample_when_predict**: True
- **hidden_channels**: 8
- **num_layers**: 2
- **num_MLP_layers**: 2
- **dropout**: 0

### Model 6: [PNA-PyG-trans]
- **base_model**: "PNA_PyG"
- **framework**: "transductive"
- **sampling_strategy**: "None"
- **lr**: 5e-4
- **weight_decay**: 1e-3
- **hidden_channels**: 64
- **num_layers**: 2
- **dropout**: 0.7
- **jk**: "cat"


### Model 7: [PNA-benchmark-trans]
- **base_model**: "PNA_Custom"
- **framework**: "transductive"
- **sampling_strategy**: "SAGE"
- **lr**: 5e-4
- **weight_decay**: 1e-3
- **hidden_channels**: 64
- **num_layers**: 2
- **num_neighbors**: [25, 10]
- **dropout**: 0.6
<!-- - **jk**: "cat" -->

### Model 8: [GINe-in]
- **base_model**: "GINe"
- **framework**: "inductive"
- **sampling_strategy**: "SAGE"
- **lr**: 5e-3
- **weight_decay**: 0
- **weighted_CE**: True
- **CE_weight**: [1, 6]
- **num_epochs**: 200
- **patience**: 20
- **batch_size**: 8192
- **add_time_stamp**: True
- **add_egoID**: True
- **add_port**: True
- **add_time_delta**: True
- **batch_norm**: True
- **seed**: 1
- **criterion**: "loss"
- **ibm_split**: True
- **f1_average**: "binary"
- **sample_when_predict**: True

- **hidden_channels**: 64
- **num_layers**: 2
- **num_neighbors**: [100, 100]
- **edge_update**: True
- **dropout**: 0.1
- **batch_norm**: "will be overwritten"
- **reverse_mp**: True
- **layer_mix**: None
- **model_mix**: "Mean"
- **skip_connection**: True

### Model 9: [PNAe-in]
- **base_model**: "PNAe"
- **framework**: "inductive"
- **sampling_strategy**: "SAGE"
- **lr**: 5e-3
- **weight_decay**: 0
- **weighted_CE**: True
- **CE_weight**: [1, 6]
- **num_epochs**: 200
- **patience**: 20
- **batch_size**: 8192
- **add_time_stamp**: True
- **add_egoID**: True
- **add_port**: True
- **add_time_delta**: True
- **batch_norm**: True
- **seed**: 1
- **criterion**: "loss"
- **ibm_split**: True
- **f1_average**: "binary"

- **hidden_channels**: 65
- **num_layers**: 2
- **num_neighbors**: [100, 100]
- **edge_update**: True
- **dropout**: 0.1
- **batch_norm**: "will be overwritten"
- **reverse_mp**: False
- **layer_mix**: None
- **model_mix**: "Mean"

 

<!-- ## Usage

To use these datasets, follow the instructions below:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/dataset_collections.git
    ```

2. Navigate to the dataset directory:
    ```bash
    cd dataset_collections
    ```

3. Load the dataset in your project:
    ```python
    import torch
    from torch_geometric.data import Dataset

    class YourDataset(Dataset):
        def __init__(self, root, transform=None, pre_transform=None):
            super(YourDataset, self).__init__(root, transform, pre_transform)
            # Add your dataset initialization code here

        @property
        def raw_file_names(self):
            return ['file1', 'file2', ...]

        @property
        def processed_file_names(self):
            return ['data.pt']

        def download(self):
            # Download to `self.raw_dir`
            pass

        def process(self):
            # Read data into huge `Data` list
            pass

        def len(self):
            return len(self.data)

        def get(self, idx):
            return self.data[idx]

    dataset = YourDataset(root='path/to/dataset')
    ```

## Contributing

If you would like to contribute to this repository, please follow the guidelines below:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch Geometric
- [Your Other References]
 -->
