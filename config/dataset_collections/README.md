# Dataset Collections

This repository contains a collection of datasets, each with its own configuration settings and parameters. Below is a detailed description of each dataset, including the task type, number of node features, and number of classes.

## Datasets

### Dataset 1: [Cora]
- **task_type**: [single-label-NC]
- **num_node_features**: [1433]
- **num_classes**: [7]

### Dataset 2: [CiteSeer]
- **task_type**: [single-label-NC]
- **num_node_features**: [3703]
- **num_classes**: [6]

### Dataset 3: [PubMed]
- **task_type**: [single-label-NC]
- **num_node_features**: [500]
- **num_classes**: [3]

### Dataset 4: [PPI]
- **task_type**: [multi-label-NC]
- **num_node_features**: [50]
- **num_classes**: [121]

### Dataset 5: [AMLworld]
- **task_type**: [single-label-EC]
- **num_classes**: [2]
- **add_time_stamp**: [True]
- **add_egoID**: [True]
- **add_port**: [True]
- **add_time_delta**: [False]
- **ibm_split**: [True]
- **force_reload**: [False]

The AMLworld datasets has 6 sub-datasets: "AMLworld-HI-Small", "AMLworld-HI-Medium", "AMLworld-HI-Large", "AMLworld-LI-Small", "AMLworld-LI-Medium", "AMLworld-LI-Large". All the datasets hsve the same configuration as AMLworld. 

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
