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