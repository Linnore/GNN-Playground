import copy

from torch_geometric.nn.conv import PNAConv

from loguru import logger
from .GraphSAGE import GraphSAGE_PyG
from .GAT import GAT_PyG, GAT_Custom
from .GIN import GIN_PyG, GIN_Custom
from .PNA import PNA_PyG, PNA_Custom


def filter_config_for_archive(config):
    archive_config = {
        "general_config": copy.deepcopy(config["general_config"]),
        "model_config": copy.deepcopy(config["model_config"]),
        "dataset_config": copy.deepcopy(config["dataset_config"]),
        "hyperparameters": copy.deepcopy(config["hyperparameters"]),
    }

    return archive_config


def get_model(config, train_loader):
    archive_config = filter_config_for_archive(config)

    model_config = config["model_config"]
    model_config.pop("num_neighbors", -1)

    dataset_config = config["dataset_config"]

    match model_config.pop("base_model"):
        case "GraphSAGE_PyG":
            model = GraphSAGE_PyG(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config.pop("hidden_node_channels"),
                num_layers=model_config.pop("num_layers"),
                dropout=model_config.pop("dropout", 0),
                jk=model_config.pop("jk", None),
                config=archive_config,
                **model_config,
            )
        case "GAT_PyG":
            model = GAT_PyG(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config.pop("hidden_node_channels"),
                num_layers=model_config.pop("num_layers"),
                heads=model_config.pop("heads", 8),
                dropout=model_config.pop("dropout", 0),
                v2=model_config.pop("v2", False),
                jk=model_config.pop("jk", None),
                config=archive_config,
                **model_config,
            )
        case "GAT_Custom":
            model = GAT_Custom(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_node_channels_per_head=model_config.pop(
                    "hidden_node_channels_per_head"),
                num_layers=model_config.pop("num_layers"),
                heads=model_config.pop("heads", 8),
                output_heads=model_config.pop("output_heads", 1),
                dropout=model_config.pop("dropout", 0),
                v2=model_config.pop("v2", False),
                jk=model_config.pop("jk", None),
                config=archive_config,
                **model_config
            )
        case "GIN_PyG":
            model = GIN_PyG(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config.pop("hidden_node_channels"),
                num_layers=model_config.pop("num_layers"),
                dropout=model_config.pop("dropout", 0),
                jk=model_config.pop("jk", None),
                config=archive_config,
                **model_config
            )
        case "GIN_Custom":
            model = GIN_Custom(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_node_channels=model_config.pop("hidden_node_channels"),
                num_layers=model_config.pop("num_layers"),
                dropout=model_config.pop("dropout", 0),
                GINE=model_config.pop("GINE", False),
                jk=model_config.pop("jk", None),
                config=archive_config,
                **model_config
            )

        case "PNA_PyG":
            deg = PNAConv.get_degree_histogram(train_loader)
            model = PNA_PyG(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config.pop("hidden_node_channels"),
                num_layers=model_config.pop("num_layers"),
                deg=deg,
                dropout=model_config.pop("dropout", 0),
                jk=model_config.pop("jk", None),
                config=archive_config,
                **model_config
            )
        case "PNA_Custom":
            deg = PNAConv.get_degree_histogram(train_loader)
            model = PNA_Custom(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_node_channels=model_config.pop("hidden_node_channels"),
                num_layers=model_config.pop("num_layers"),
                deg=deg,
                dropout=model_config.pop("dropout", 0),
                jk=model_config.pop("jk", None),
                config=archive_config,
                **model_config
            )
        case _:
            raise NotImplementedError(
                f"Unreconized base model: {model_config['base_model']}")

    return model
