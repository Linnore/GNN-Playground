import copy

from torch_geometric.nn.conv import PNAConv

from loguru import logger  # noqa

from .GraphSAGE import GraphSAGE_PyG
from .GAT import GAT_PyG, GAT_Custom
from .GIN import GIN_PyG, GIN_Custom, GINe
from .PNA import PNA_PyG, PNA_Custom, PNAe


def filter_config_for_archive(config):
    archive_config = {
        "general_config": copy.deepcopy(config["general_config"]),
        "model_config": copy.deepcopy(config["model_config"]),
        "dataset_config": copy.deepcopy(config["dataset_config"]),
        "hyperparameters": copy.deepcopy(config["hyperparameters"]),
    }

    return archive_config


def get_readout(task_type):
    if task_type.endswith("NC"):
        return "node"
    elif task_type.endswith("EC"):
        return "edge"
    else:
        raise NotImplementedError


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
                hidden_channels=model_config.pop("hidden_channels"),
                num_layers=model_config.pop("num_layers"),
                config=archive_config,
                **model_config,
            )
        case "GAT_PyG":
            model = GAT_PyG(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config.pop("hidden_channels"),
                num_layers=model_config.pop("num_layers"),
                heads=model_config.pop("heads", 8),
                v2=model_config.pop("v2", False),
                config=archive_config,
                **model_config,
            )
        case "GAT_Custom":
            model = GAT_Custom(in_channels=dataset_config["num_node_features"],
                               out_channels=dataset_config["num_classes"],
                               hidden_channels_per_head=model_config.pop(
                                   "hidden_channels_per_head"),
                               num_layers=model_config.pop("num_layers"),
                               heads=model_config.pop("heads", 8),
                               output_heads=model_config.pop(
                                   "output_heads", 1),
                               v2=model_config.pop("v2", False),
                               config=archive_config,
                               **model_config)
        case "GIN_PyG":
            model = GIN_PyG(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config.pop("hidden_channels"),
                num_layers=model_config.pop("num_layers"),
                config=archive_config,
                **model_config)
        case "GIN_Custom":
            model = GIN_Custom(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config.pop("hidden_channels"),
                num_layers=model_config.pop("num_layers"),
                GINE=model_config.pop("GINE", False),
                config=archive_config,
                **model_config)

        case "PNA_PyG":
            deg = PNAConv.get_degree_histogram(train_loader)
            model = PNA_PyG(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config.pop("hidden_channels"),
                num_layers=model_config.pop("num_layers"),
                deg=deg,
                config=archive_config,
                **model_config)
        case "PNA_Custom":
            deg = PNAConv.get_degree_histogram(train_loader)
            model = PNA_Custom(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config.pop("hidden_channels"),
                num_layers=model_config.pop("num_layers"),
                deg=deg,
                config=archive_config,
                **model_config)
        case "GINe":
            model_config["readout"] = get_readout(dataset_config["task_type"])
            model = GINe(in_channels=dataset_config["num_node_features"],
                         out_channels=dataset_config["num_classes"],
                         hidden_channels=model_config.pop("hidden_channels"),
                         edge_dim=dataset_config["num_edge_features"],
                         num_layers=model_config.pop("num_layers"),
                         edge_update=model_config.pop("edge_update", False),
                         batch_norm=model_config.pop("batch_norm", True),
                         config=archive_config,
                         **model_config)
        case "PNAe":
            deg = PNAConv.get_degree_histogram(train_loader)
            model_config["readout"] = get_readout(dataset_config["task_type"])
            model = PNAe(in_channels=dataset_config["num_node_features"],
                         out_channels=dataset_config["num_classes"],
                         hidden_channels=model_config.pop("hidden_channels"),
                         edge_dim=dataset_config["num_edge_features"],
                         num_layers=model_config.pop("num_layers"),
                         edge_update=model_config.pop("edge_update", False),
                         batch_norm=model_config.pop("batch_norm", True),
                         deg=deg,
                         config=archive_config,
                         **model_config)
        case _:
            raise NotImplementedError(
                f"Unreconized base model: {model_config['base_model']}")

    return model
