import torch
import copy

from loguru import logger
from .model.GraphSAGE import GraphSAGE_DGL
from .model.GAT import GAT_DGL_Custom
from .model.GIN import GIN_DGL_Custom

def get_class_pos_weights(dataset_config, train_loader):
    total_num = 0
    pos_cnt = torch.zeros(dataset_config["num_classes"], dtype=int)
    for batch in train_loader:
        total_num += batch.y.shape[0]
        pos_cnt = pos_cnt + batch.y.sum(axis=0)
    neg_cnt = total_num - pos_cnt
    return neg_cnt/pos_cnt
        

def get_loss_fn(config, train_loader, reduction="sum"):
    dataset_config = config["dataset_config"]
    if dataset_config["task_type"] == "single-label-NC":
        return torch.nn.CrossEntropyLoss(reduction=reduction)
    
    elif dataset_config["task_type"] == "multi-label-NC":
        if config["hyperparameters"]["weighted_BCE"]:
            pos_weight = get_class_pos_weights(dataset_config, train_loader)
        else:
            pos_weight = torch.ones(dataset_config["num_classes"])
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
    else:
        logger.exception(NotImplementedError("Unsupported task type!"))


def filter_config_for_archive(config):
    archive_config = {
        "general_config": copy.deepcopy(config["general_config"]),
        "model_config": copy.deepcopy(config["model_config"]),
        "dataset_config": copy.deepcopy(config["dataset_config"]),
        "hyperparameters": copy.deepcopy(config["hyperparameters"]),
    }
    
    return archive_config
    

def get_model(config):
    archive_config = filter_config_for_archive(config)
    
    model_config = config["model_config"]
    model_config.pop("num_neighbors", -1)

    dataset_config = config["dataset_config"]

    match model_config.pop("base_model"):
        case "GraphSAGE_DGL":
            model = GraphSAGE_DGL(
                in_channels=dataset_config["num_node_features"],
                hidden_channels=model_config.pop("hidden_node_channels"),
                out_channels=dataset_config["num_classes"],
                num_layers=model_config.pop("num_layers"),
                dropout=model_config.pop("dropout", 0),
                aggregator_type="mean",
                activation="relu",
                norm=None,
                jk=model_config.pop("jk", None),
                config=archive_config,
                **model_config,
            )
        case "GAT_DGL_Custom":
            model = GAT_DGL_Custom(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_node_channels_per_head=model_config.pop("hidden_node_channels_per_head"),
                num_layers=model_config.pop("num_layers"),
                heads=model_config.pop("heads", 8),
                output_heads=model_config.pop("output_heads", 1),
                dropout=model_config.pop("dropout", 0),
                attention_dropout=model_config.pop("attention_dropout", 0),
                v2=model_config.pop("v2", False),
                residual=model_config.pop("residual", True),
                # jk=model_config.pop("jk", None),
                # skip_connection=config.pop("skip_connection", False),
                config=archive_config,
                **model_config,
            )
        case "GIN_DGL_Custom":
            model = GIN_DGL_Custom(
                input_dim=dataset_config["num_node_features"],
                hidden_dim=model_config.pop("hidden_node_channels"),
                output_dim=dataset_config["num_classes"],
                num_lin_layers=model_config.pop("num_lin_layers"),
                num_gcn_layers=model_config.pop("num_gcn_layers"),
                aggregator_type=model_config.pop("aggregator_type"),
                init_eps=model_config.pop("init_eps"),
                learn_eps=model_config.pop("learn_eps"),
                mlp_activation='relu',
                gin_activation='relu',
                dropout=model_config.pop("dropout"),
                mlp_dropout=model_config.pop("mlp_dropout"),
                mlp_bias=model_config.pop("mlp_bias"),
                config=archive_config,
                **model_config,
            )
        case _:
            logger.exception(
                f"Unreconized base model: {model_config['base_model']}")

    return model