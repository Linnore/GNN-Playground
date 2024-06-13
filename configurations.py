"""
Configurations
"""


class config:

    mode = None  # Will be overwritten as train/inference by command line.
    model = None  # Must be one of the model in model_collections
    dataset = None  # Must be one of the dataset in dataset_collections

    # MLflow tracking server configrations
    mlflow_config = {
        "tracking_uri": "http://127.0.0.1:8080/",
        "username": "admin",
        "password": "password",
        "experiment": "Default",
        "auth": True
    }

    general_config = {
        "framework": "inductive",
        "seed": 118010142,
        "device": "cpu",
        "tqdm": False,
        "save_model": True,
        "criterion": "loss",
        "num_epochs": 300,
        "patience": 15,
        "num_workers": 0,
    }

    # Hyperparameters
    hyperparameters = {
        "inductive_type": "SAGE",  # Must be choosen from inductive_type options

        # Used if inductive_type is SAGE; Must be choosen from SAGE_options
        "SAGE_option": "strict",

        "batch_size": 64,
        "lr": 1e-4
    }

    """Options for difference experiment settings

    transductive: 
        The data split will follow a transductive manner. Training nodes can connected with validation nodes or testing nodes, though backward propogation is only applied on the loss computed by the training nodes.
        
    inductive:
        - If inductive_type is "SAGE":
            -  default or strict
                The data will be split to train_subgraph, val_subgraph, and test_subgraph. No message passing across the train. val, and test datasets is allowed.
            -  soft:
                Training nodes are cut off from validation nodes and testing nodes to form a training subgraph. However, when doing inference on the validation nodes and testing nodes, the edges  <Node_train, Node_val/Node_test>  and <Node_val, Node_test> can be used.
        - If inductive_type is "SAINT"
            TODO
    """
    framework_options = [
        "transductive",
        "inductive"
    ]
    inductive_type_options = [
        "SAGE",  # use the inductive learning proposed by GraphSAGE
        "SAINT"  # use the inductive learning proposed by GraphSAINT
    ]
    SAGE_options = [
        "default",
        "strict",
        "soft",
    ]

    # Model collections. Create the configuration of each model here.
    model_collections = {
        "GraphSAGE-mean": {
            "base_model": "GraphSAGE_PyG",
            "num_layers": 2,
            "hidden_node_channels": 256,
            "num_neighbors": [25, 10],
            "dropout": 0,
            "jk": None
        },


        "GAT": {
            "base_model": "GAT_PyG",
            "num_layers": 2,
            "hidden_node_channels": 256,
            "num_neighbors": [25, 10],
            "dropout": 0,
            "jk": None,
            "v2": False
        },


        "GIN": {

        }
    }

    # Dataset collections
    dataset_collections = {
        "Cora": {
            "num_node_features": 1433,
            "num_classes": 7
        },
        "CiteSeer": {
            "num_node_features": 3703,
            "num_classes": 6
        },
        "PubMed": {
            "num_node_features": 500,
            "num_classes": 3
        },
        "Reddit": {
            "num_node_features": 602,
            "num_classes": 41
        },
        "Reddit2": {
            "num_node_features": 602,
            "num_classes": 41
        },
        "Flickr": {
            "num_node_features": 500,
            "num_classes": 7
        },
        "Yelp": {
            "num_node_features": 300,
            "num_classes": 100
        },
        "AmazonProducts": {
            "num_node_features": 200,
            "num_classes": 107
        }

    }
