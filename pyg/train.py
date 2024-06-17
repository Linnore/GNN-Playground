import torch
import mlflow
import pprint
import os
import copy

from .data_utils import get_loader
from .model.GraphSAGE import GraphSAGE_PyG
from .model.GAT import GAT_PyG, GAT_Custom

from loguru import logger
from torch_geometric.nn import summary
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score


def get_model(config):
    model_config = config["model_collections"][config["model"]]
    dataset_config = config["dataset_collections"][config["dataset"]]
    match model_config.pop("base_model"):
        case "GraphSAGE_PyG":
            model = GraphSAGE_PyG(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config.pop("hidden_node_channels"),
                num_layers=model_config.pop("num_layers"),
                dropout=model_config.pop("dropout", 0),
                jk=model_config.pop("jk", None),
                **model_config,
            )
        case "GAT_PyG":
            model = GAT_PyG(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config.pop("hidden_node_channels"),
                num_layers=model_config.pop("num_layers"),
                dropout=model_config.pop("dropout", 0),
                jk=model_config.pop("jk", None),
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
                v2=model_config.pop("v2", None),
                **model_config
            )

        case "GIN_PyG":
            pass
        case _:
            logger.exception(
                f"Unreconized base model: {model_config['base_model']}")

    return model


def run_step(mode: str, epoch, model, loss_fn, optimizer, loader, general_config):
    if mode == "test":
        model.eval()
    else:
        total_loss = 0
        total_num = 0
        if mode == "train":
            model.train()
        elif mode == "val":
            model.eval()

    predictions = []
    truths = []
    bar = tqdm(loader, total=len(loader), disable=not general_config["tqdm"])
    for batch in bar:
        if mode == "train":
            optimizer.zero_grad()

        targets = batch.y[:batch.batch_size]
        outputs = model(batch.x, batch.edge_index)[:batch.batch_size]

        predictions.append(outputs.argmax(dim=-1))
        truths.append(targets)

        if mode != "test":
            loss = loss_fn(outputs, targets)
            loss.backward()

            if mode == "train":
                optimizer.step()

            loss = loss.detach().cpu()
            num_targets = outputs.numel()
            total_loss += loss * num_targets
            total_num += num_targets
            bar.set_description(f"{mode}_loss={loss:<8.6g}")

    # Metrics
    predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
    truths = torch.cat(truths, dim=0).detach().cpu().numpy()

    if mode != "test":
        avg_loss = total_loss/total_num
        mlflow.log_metric(f"{mode} loss", avg_loss, epoch)

    f1 = f1_score(truths, predictions, average="micro")
    mlflow.log_metric(f"{mode} F1", f1, epoch)

    if mode == "train":
        return loss, f1
    elif mode == "val":
        return loss, f1
    elif mode == "test":
        return f1, truths, predictions


def train_gnn(config):

    general_config = config["general_config"]
    device = general_config["device"]

    # Initialize MLflow Logging
    run_name = f"{config['model']}"
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tag(
        "base model", config["model_collections"][config["model"]]["base_model"])
    mlflow.set_tag("dataset", config["dataset"])
    logger.info(f"Launching run: {run.info.run_name}")

    # Log hyperparameters
    params = config["hyperparameters"]
    params.update(config["model_collections"][config["model"]])
    params_str = pprint.pformat(params)
    general_config_str = pprint.pformat(general_config)
    logger.info(f"General configurations:\n{general_config_str}")
    logger.info(f"Hyperparameters:\n{params_str}")
    mlflow.log_params(general_config)
    mlflow.log_params(params)

    # Get loaders
    train_loader, val_loader, test_loader = get_loader(config)

    # Get model
    model = get_model(config)
    model.to(device).reset_parameters()

    # Setup save directory for optimizer states
    if general_config["save_model"]:
        save_path = os.path.join(
            "logs/tmp", f"{run.info.run_name}-Optimizer-{run.info.run_id}.tar")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

    # Summary logging
    sample_batch = next(iter(train_loader))
    sample_x = sample_batch.x
    sample_edge_index = sample_batch.edge_index
    summary_str = summary(model, sample_x, sample_edge_index)
    logger.info("Model Summary:\n" + summary_str)
    with open("logs/tmp/model_summary.txt", "w") as out_file:
        out_file.write(summary_str)
    mlflow.log_artifact("logs/tmp/model_summary.txt")

    # Setup Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    # Setup loss function
    logger.warning('Todo: Implement WCE.')
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    # Setup metrics
    criterion = general_config["criterion"].lower()
    if criterion == "loss":
        best_value = -2147483647
    elif criterion == "f1":
        best_value = -1

    # Training loop
    patience = general_config["patience"]
    if patience == None:
        patience = general_config["num_epochs"]
        
    
    best_epoch = 0
    for epoch in range(1, 1+general_config["num_epochs"]):
        if general_config["tqdm"]:
            print(f"Epoch {epoch}:")
        # Batch training
        train_loss, train_f1= run_step(
            "train", epoch, model, loss_fn, optimizer, train_loader, general_config)

        # Validation
        val_loss, val_f1 = run_step(
            "val", epoch, model, loss_fn, optimizer, val_loader, general_config)

        # Test
        test_f1, truths, predictions = run_step(
            "test", epoch, model, loss_fn, optimizer, test_loader, general_config)

        logger.info(f"Epoch {epoch}: train_loss={train_loss:<8.6g}, train_f1={train_f1:<8.6g}, val_loss={val_loss:<8.6g}, val_f1={val_f1:<8.6g}, test_f1={test_f1:<8.6g}")

        # Best model
        if criterion == "loss":
            criterion_value = -val_loss
        elif criterion == "f1":
            criterion_value = val_f1

        if criterion_value > best_value:
            best_value = criterion_value
            mlflow.log_metric("Best Test F1", test_f1, epoch)

            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_report = classification_report(
                truths, predictions, zero_division=0)
            best_epoch = epoch

            if general_config["save_model"]:
                torch.save(
                    {
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                    },
                    save_path
                )

        # Early Stopping
        if epoch-best_epoch > patience:
            logger.info("Patience reached. Early stop the trainning.")
            break

    model.load_state_dict(best_model_state_dict)
    if general_config["save_model"]:
        mlflow.pytorch.log_model(model, "Best Model")
        mlflow.log_artifact(save_path, "Optimizer States")
        os.remove(save_path)

    with open("logs/tmp/test_report.txt", "w") as out_file:
        out_file.write(best_report)
    mlflow.log_artifact("logs/tmp/test_report.txt")

    logger.info(f"Best model report:\n{best_report}")

    mlflow.end_run()
