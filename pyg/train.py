import torch
import mlflow
import pprint
import os
import copy

import numpy as np

from .data_utils import get_loader
from .model.model_hub import get_model
from .train_utils import get_loss_fn, get_run_step, get_batch_input

from loguru import logger
from torch_geometric.nn import summary
from sklearn.metrics import classification_report

from mlflow import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


def train_gnn(config):
    mlflow_config = config["mlflow_config"]
    general_config = config["general_config"]
    device = general_config["device"]
    dataset_config = config["dataset_config"]
    model_config = config["model_config"]
    register_info = model_config.pop("register_info", {})

    # Initialize MLflow Logging
    logger.info(f"Launching experiment: {mlflow_config['experiment']}")
    mlflow.set_experiment(mlflow_config["experiment"])
    run_name = f"{config['model']}-{config['dataset']}"
    model_name = run_name
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tag(
        "base model", model_config["base_model"])
    mlflow.set_tag("dataset", config["dataset"])
    logger.info(f"Launching run: {run.info.run_name}")

    # Log hyperparameters
    params = config["hyperparameters"]
    params.update(model_config)
    params_str = pprint.pformat(params)
    general_config_str = pprint.pformat(general_config)
    logger.info(f"General configurations:\n{general_config_str}")
    logger.info(f"Hyperparameters:\n{params_str}")
    mlflow.log_params(general_config)
    mlflow.log_params(params)

    # Get loaders
    train_loader, val_loader, test_loader = get_loader(config)

    # Get model
    model = get_model(config, train_loader).to(device)
    model.reset_parameters()

    # Setup loss function
    loss_fn = get_loss_fn(config, train_loader, reduction='mean')
    logger.info(f"Loss function: {loss_fn}")

    # Setup save directory for optimizer states
    save_path = os.path.join(
        "logs/tmp", f"{run.info.run_name}-Optimizer-{run.info.run_id}.tar")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # Summary logging
    reverse_mp = model_config.get("reverse_mp", False)
    sample_input = get_batch_input(next(iter(train_loader)), reverse_mp, device)
    summary_str = summary(model, **sample_input)
    logger.info("Model Summary:\n" + summary_str)
    with open("logs/tmp/model_summary.txt", "w") as out_file:
        out_file.write(summary_str)
    mlflow.log_artifact("logs/tmp/model_summary.txt")

    # Setup Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    # Setup metrics
    criterion = general_config["criterion"].lower()
    if criterion == "loss":
        best_value = -2147483647
    elif criterion == "f1":
        best_value = -1

    # Training loop
    patience = general_config["patience"]
    if patience == None or patience <= 0:
        patience = general_config["num_epochs"]

    # Setup training steps according to task type
    run_step = get_run_step(
        model,
        loss_fn,
        optimizer,
        sampling_strategy=config["general_config"]["sampling_strategy"],
        enable_tqdm=general_config["tqdm"],
        device=device,
        task_type=dataset_config["task_type"], 
        reverse_mp=reverse_mp,
        f1_average=general_config["f1_average"])

    best_epoch = 0
    for epoch in range(1, 1+general_config["num_epochs"]):
        if general_config["tqdm"]:
            print(f"Epoch {epoch}:")

        # Training
        model.train()
        train_loss, train_f1, _, _ = run_step("train", epoch, train_loader)

        with torch.no_grad():
            # Validation
            model.eval()
            val_loss, val_f1, _, _ = run_step("val", epoch, val_loader)

            # Test
            _, test_f1, predictions, truths = run_step(
                "test", epoch, test_loader)

        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:<8.6g}, train_f1={train_f1:<8.6g}, val_loss={val_loss:<8.6g}, val_f1={val_f1:<8.6g}, test_f1={test_f1:<8.6g}")

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

    # Save model
    model.load_state_dict(best_model_state_dict)
    input_schema = Schema(
        [
            TensorSpec(np.dtype(np.float32),
                       (-1, dataset_config["num_node_features"]), "x"),
            TensorSpec(np.dtype(np.int64), (2, -1), "edge_index")
        ]
    )
    output_schema = Schema(
        [TensorSpec(np.dtype(np.float32), (-1, dataset_config["num_classes"]))])

    mlflow.pytorch.log_model(model,
                             model_name,
                             signature=ModelSignature(
                                 inputs=input_schema, outputs=output_schema),
                             )
    mlflow.log_artifact(save_path, "Optimizer States")
    os.remove(save_path)

    # Register the model
    if general_config["register_model"]:
        model_uri = f"runs:/{run.info.run_id}/{model_name}"
        reg_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=register_info.get("tags", {})
        )

        # Write inference instructions as description
        desc = [register_info.get("description")] if register_info.get(
            "description") else []
        desc.extend([
            "Use the following codes for inference:",
            "```python",
            "from mlflow.pytorch import load_model as load_pyt_model",
            f"loaded_model = load_pyt_model('{model_uri}')",
            "output = loaded_model(**input)",
            "```"
        ])
        desc = "\n".join(desc)

        client = MlflowClient()
        client.update_model_version(
            name=model_name,
            version=reg_model.version,
            description=desc
        )
        logger.success(
            f"Model registered as {model_name}, version {reg_model.version}")

    # Save report
    logger.info(f"Best model report:\n{best_report}")
    with open("logs/tmp/test_report.txt", "w") as out_file:
        out_file.write(best_report)
    mlflow.log_artifact("logs/tmp/test_report.txt")

    mlflow.end_run()
