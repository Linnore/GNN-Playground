"""
Configurations

Priority of configuration:
Command line arguments >
configuraions in model["overwirte"] >
other configuration
"""


class config:

    mode = None  # Will be overwritten as train/inference by command line.
    model = None  # Must be one of the model in model_collections
    dataset = None  # Must be one of the dataset in dataset_collections
    dataset_dir = "./dataset"  # Local directory to save datasets.

    # MLflow tracking server configrations
    mlflow_config = {
        "tracking_uri": "http://127.0.0.1:8080/",
        "username": "admin",
        "password": "password",
        "experiment": "Default",
        "auth": True
    }

    general_config = {
        "framework": "transductive",  # Must be transductive or inductive
        "sampling_strategy":
        "None",  # Must be choosen from sampling_strategy options

        # Used if sampling_strategy is SAGE;
        # Must be choosen from SAGE_inductive_options
        "SAGE_inductive_option": "strict",

        # Enable to use sampling strategy
        # when predicting (val, test, inference). Default: False.
        "sample_when_predict": False,
        "seed": 118010142,
        "device": "cpu",
        "tqdm": False,
        "verbose": False,
        "register_model": False,
        "criterion": "loss",
        "num_epochs": 1000,
        "patience": 100,
        "num_workers": 2,
        "persistent_workers": True,
        "f1_average": "micro",
    }

    # Hyperparameters
    hyperparameters = {
        # batch_size to infinity if memory allowed
        "batch_size": 512,
        "lr": 5e-3,
        "weight_decay": 0.0005,  # L2 regularization,
        "weighted_CE": False,  # Only useful for single-label task.
        "CE_weight": [1, 6],
        "weighted_BCE": False,  # Only useful for multi-label task.
    }
    """Options for difference experiment settings
    `framework`
        `"transductive"`:
            The data split will follow a transductive manner.
            All edges are presented in any phase (train; val; test). However,
            only the training nodes/edges can be used to compute the loss value
            and do backward propergation in the training phase.

        `"inductive"`:
            - If `sampling_strategy` is `"SAGE"`,
                `SAGE_inductive_option`:
                -  `"default"` or `"strict"`
                    The data will be split to train_subgraph, val_subgraph,
                    and test_subgraph. No message passing across the train,
                    val, and test datasets is allowed.
                -  `soft`:
                    The train_subgraph only contains the training nodes/edges
                    as in the strict option. However, the val_subgraph includes
                    the training nodes & edges, validation nodes & edges, and
                    messages between the training nodes and validation nodes,
                    which means train_subgraph is a proper subgraph of the
                    val_subgraph. Similarly, test_subgraph contains
                    train_subgraph, val_subgraph, and all messages among
                    training nodes, validation nodes, and testing nodes.
                    For each of the above subgraphs, train_mask, val_mask,
                    and test_mask should be provided.
                    E.g., val.train_mask is 0 for all nodes/edges.
                    val_subgraph.val_mask is 1 for all validation nodes/edges,
                    and is 0 for all training and testing nodes/edges.
                    val_subgraph.test_mask is 0 for all nodes/edges.

            - If `sampling_strategy` is `"SAINT"`
                TODO

            - If `sampling_strategy` is `"None"`. string(`"None"`)!!!
                One batch for each split. All neighbors will be used,
                so the model behaves like GCN.

            - If `sampling_strategy` is `"GraphBatching"`:
                Graph-level batching. This option is useful for dataset
                containing multiple graphs. Each graph will be regarded
                as one batch-element as a whole. E.g., batch_size = 2 will give
                each batch containing 2 graphs.
    """
    framework_options = ["transductive", "inductive"]
    sampling_strategy_options = [
        "SAGE",  # use the inductive learning proposed by GraphSAGE
        "SAINT",  # use the inductive learning proposed by GraphSAINT
        "GraphBatching",  # use for inductive learning with full graph
        "None",  # only useful when GCN-like behaviors are desired.
    ]
    SAGE_inductive_options = [
        "default",
        "strict",
        "soft",
    ]

    # Model collections. Create the configuration of each model here.
    model_collections = {
        "GraphSAGE-benchmark-trans": {
            "base_model": "GraphSAGE_PyG",
            "overwrite": {
                "framework": "transductive",
                "sampling_strategy": "SAGE",
                "lr": 5e-3,
                "weight_decay": 5e-4,
            },
            # Registeration inforamtion for MLflow
            "register_info": {
                "description": "Benchmark GraphSAGE.",
                "tags": {
                    "aggr": "mean",
                },
            },
            "num_layers": 2,
            "num_neighbors": [25, 10],
            "hidden_channels": 64,
            "dropout": 0,
            "aggr": "mean",
        },

        # Transductive benchmark GAT of Planetoid;
        # For PubMed: output_heads=8, lr=0.01, weight_decay=0.001
        "GAT-benchmark-trans": {
            "base_model": "GAT_Custom",
            "overwrite": {
                "framework": "transductive",
                "sampling_strategy": "None",
                "lr": 5e-3,
                "weight_decay": 5e-4,
            },
            # Registeration inforamtion for MLflow
            "register_info": {
                "description": "Benchmark model from GAT paper.",
                "tags": {
                    "v2": False,  # Change to v2 if v2 is True
                },
            },
            "hidden_channels_per_head": 32,
            "num_layers": 2,
            "heads": 8,
            "output_heads": 1,
            "dropout": 0.6,
            "jk": None,
            "v2": False,
        },

        # Inductive benchmark GAT of PPI;
        "GAT-benchmark-in": {
            "base_model": "GAT_Custom",
            "overwrite": {
                "framework": "inductive",
                "sampling_strategy": "GraphBatching",
                "batch_size": 2,
                "lr": 5e-3,
                "weight_decay": 0,
            },
            # Registeration inforamtion for MLflow
            "register_info": {
                "description": "Benchmark model from GAT paper.",
                "tags": {
                    "v2": False,  # Change to v2 if v2 is True
                },
            },
            "num_layers": 3,
            "heads": [4, 4],
            "hidden_channels_per_head": [256, 256],
            "output_heas": 6,
            "dropout": 0,
            "jk": None,
            "v2": False,
            "skip_connection": True,
        },

        # # Equivalent to GAT with equal attention + jk.
        # # TODO: Customizable GraphSAGE with skip-connection
        # "GAT-Equal-benchmark-in":{
        #     "base_model": "GraphSAGE_PyG",
        #     "overwrite": {
        #         "framework": "inductive",
        #         "sampling_strategy": "GraphBatching",
        #         "batch_size": 2,
        #         "lr" : 5e-3,
        #         "weight_decay": 0,
        #     },
        #     "num_layers": 3,
        #     "hidden_channels": 1024,
        #     "dropout": 0,
        #     "aggr": "mean",
        #     "jk": "cat"
        # },

        # Current GIN implementation has low acc for the node-level task.
        "GIN-benchmark-trans": {
            "base_model": "GIN_Custom",
            "overwrite": {
                "framework": "transductive",
                "sampling_strategy": "None",
                "lr": 5e-3,
                "weight_decay": 5e-4,
            },
            # Registeration inforamtion for MLflow
            "register_info": {
                "description": "Benchmark GIN model.",
                "tags": {
                    "GINE": False,  # True if use GINEConv
                },
            },
            "hidden_channels": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "jk": "cat",
            "GINE": False,
            "skip_connection": False,
        },
        "GIN-PyG-trans": {
            "base_model": "GIN_PyG",
            "overwrite": {
                "framework": "transductive",
                "sampling_strategy": "None",
                "lr": 1e-2,
                "weight_decay": 5e-4,
            },
            # Registeration inforamtion for MLflow
            "register_info": {
                "description": "Benchmark GIN model.",
                "tags": {
                    "GINE": False,  # True if use GINEConv
                },
            },
            "hidden_channels": 8,
            "num_layers": 2,
            "num_MLP_layers": 2,
            "dropout": 0,
        },
        "PNA-PyG-trans": {
            "base_model": "PNA_PyG",
            "overwrite": {
                "framework": "transductive",
                "sampling_strategy": "None",
                "lr": 5e-4,
                "weight_decay": 1e-3,
            },
            # Registeration inforamtion for MLflow
            "register_info": {
                "description": "Benchmark PNA model.",
                "tags": {},
            },
            "hidden_channels": 64,
            "num_layers": 2,
            "dropout": 0.7,
            "jk": "cat"
        },
        "PNA-benchmark-trans": {
            "base_model": "PNA_Custom",
            "overwrite": {
                "framework": "transductive",
                "sampling_strategy": "SAGE",
                "lr": 5e-4,
                "weight_decay": 1e-3,
            },
            # Registeration inforamtion for MLflow
            "register_info": {
                "description": "Benchmark PNA model.",
                "tags": {},
            },
            "hidden_channels": 64,
            "num_layers": 2,
            "num_neighbors": [25, 10],
            "dropout": 0.6,
            # "jk": "cat"
        },
        "GINe-in": {
            "base_model": "GINe",
            "overwrite": {
                "framework": "inductive",
                "sampling_strategy": "SAGE",
                "lr": 5e-3,
                "weight_decay": 0,
                "weighted_CE": True,
                "CE_weight": [1, 6],
                "num_epochs": 200,
                "patience": 20,
                "batch_size": 8192,
                "add_time_stamp": True,
                "add_egoID": True,
                "add_port": True,
                "add_time_delta": False,
                "batch_norm": True,
                "seed": 1,
                "criterion": "loss",
                "ibm_split": True,
                "f1_average": "binary"
            },
            "register_info": {
                "description": "GINe in IBM MultiGNN's paper.",
                "tags": {
                    "add_time_stamp": "will be overwritten",
                    "add_egoID": "will be overwritten",
                    "add_port": "will be overwritten",
                    "add_time_delta": "will be overwritten",
                    "batch_norm": "will be overwritten",
                    "ibm_split": "will be overwritten",
                }
            },
            "hidden_channels": 66,
            "num_layers": 2,
            "num_neighbors": [100, 100],
            "edge_update": True,
            "dropout": 0.1,
            "batch_norm": "will be overwritten",
            "reverse_mp": True,
            "layer_mix": "None",
            "model_mix": "Mean",
        },
        "GINe-in-NC": {
            "base_model": "GINe",
            "overwrite": {
                "framework": "inductive",
                "sampling_strategy": "SAGE",
                "lr": 5e-3,
                "weight_decay": 0,
                "weighted_CE": True,
                "CE_weight": [1, 6],
                "num_epochs": 200,
                "patience": 20,
                "batch_size": 8192,
                "add_time_stamp": True,
                "add_egoID": True,
                "add_port": True,
                "add_time_delta": False,
                "batch_norm": True,
                "seed": 118010142,
                "criterion": "loss",
                "ibm_split": True,
                "f1_average": "binary",
                "task_type": "single-label-NC",
            },
            "register_info": {
                "description": "GINe in IBM MultiGNN's paper.",
                "tags": {
                    "add_time_stamp": "will be overwritten",
                    "add_egoID": "will be overwritten",
                    "add_port": "will be overwritten",
                    "add_time_delta": "will be overwritten",
                    "batch_norm": "will be overwritten",
                    "ibm_split": "will be overwritten",
                }
            },
            "hidden_channels": 66,
            "num_layers": 2,
            "num_neighbors": [100, 100],
            "edge_update": True,
            "dropout": 0.1,
            "batch_norm": "will be overwritten",
            "reverse_mp": False,
            "layer_mix": "None",
            "model_mix": "Mean",
        },
        "PNAe-in": {
            "base_model": "PNAe",
            "overwrite": {
                "framework": "inductive",
                "sampling_strategy": "SAGE",
                "lr": 5e-3,
                "weight_decay": 0,
                "weighted_CE": True,
                "CE_weight": [1, 6],
                "num_epochs": 200,
                "patience": 20,
                "batch_size": 8192,
                "add_time_stamp": True,
                "add_egoID": True,
                "add_port": True,
                "add_time_delta": False,
                "batch_norm": True,
                "seed": 1,
                "criterion": "loss",
                "ibm_split": True,
                "f1_average": "binary"
            },
            "register_info": {
                "description": "PNAe in IBM MultiGNN's paper.",
                "tags": {
                    "add_time_stamp": "will be overwritten",
                    "add_egoID": "will be overwritten",
                    "add_port": "will be overwritten",
                    "add_time_delta": "will be overwritten",
                    "batch_norm": "will be overwritten",
                    "ibm_split": "will be overwritten",
                }
            },
            "hidden_channels": 65,
            "num_layers": 2,
            "num_neighbors": [100, 100],
            "edge_update": True,
            "dropout": 0.1,
            "batch_norm": "will be overwritten",
            "reverse_mp": False,
            "layer_mix": "None",
            "model_mix": "Mean",
        },
    }

    # Dataset collections
    # Task type:
    #  - "single-label-NC" : single label node classification
    #  - "multi-label-NC" : multi-label node classification
    task_type_options = [
        "single-label-NC",
        "multi-label-NC",
        "single-label-EC",
    ]
    dataset_collections = {
        "Cora": {
            "task_type": "single-label-NC",
            "num_node_features": 1433,
            "num_classes": 7
        },
        "CiteSeer": {
            "task_type": "single-label-NC",
            "num_node_features": 3703,
            "num_classes": 6
        },
        "PubMed": {
            "task_type": "single-label-NC",
            "num_node_features": 500,
            "num_classes": 3
        },
        "Reddit": {
            "task_type": "single-label-NC",
            "num_node_features": 602,
            "num_classes": 41
        },
        "Reddit2": {
            "task_type": "single-label-NC",
            "num_node_features": 602,
            "num_classes": 41
        },
        "Flickr": {
            "task_type": "single-label-NC",
            "num_node_features": 500,
            "num_classes": 7
        },
        "Yelp": {
            "task_type": "single-label-NC",
            "num_node_features": 300,
            "num_classes": 100
        },
        "AmazonProducts": {
            "task_type": "single-label-NC",
            "num_node_features": 200,
            "num_classes": 107
        },
        "PPI": {
            "task_type": "multi-label-NC",
            "num_node_features": 50,
            "num_classes": 121
        },

        # node_feature includes: a dummy 1, and EgoID
        "AMLworld-HI-Small": {
            # configured by AMLworld_config
        },
        "AMLworld-HI-Medium": {
            # configured by AMLworld_config
        },
        "AMLworld-HI-Large": {
            # configured by AMLworld_config
        },
        "AMLworld-LI-Small": {
            # configured by AMLworld_config
        },
        "AMLworld-LI-Medium": {
            # configured by AMLworld_config
        },
        "AMLworld-LI-Large": {
            # configured by AMLworld_config
        },
    }

    # Configuration of AMLworld dataset:
    # Supported task type: single-label-EC (default) and single-label-NC
    AMLworld_config = {
        "task_type": "single-label-EC",
        "num_classes": 2,
        "add_time_stamp": True,
        "add_egoID": True,
        "add_port": True,
        "add_time_delta": False,
        "ibm_split": True,
        "force_reload": False,
    }
