import torch
import os

from .data_utils import get_loader
from .train_utils import get_batch_input, append_source_edges

from mlflow import MlflowClient
from mlflow.pytorch import load_model as load_pyt_model

from loguru import logger
from tqdm import tqdm
from sklearn.metrics import classification_report

OVERWRITE_MESSAGE_SET = set()


def overwrite_config(config, key, value):
    global OVERWRITE_MESSAGE_SET
    if value is not None:
        if value != config[key]:
            message = f'Overwrite {key}={value} from {key}={config[key]}'
            if message not in OVERWRITE_MESSAGE_SET:
                OVERWRITE_MESSAGE_SET.add(message)
                logger.warning(
                    f'Overwrite {key}={value} from {key}={config[key]}')
        config[key] = value


def update_config(config: dict, vargs: dict):
    config["mode"] = vargs["mode"]
    config["model"] = vargs["model"]
    config["dataset"] = vargs["dataset"]

    if config["mode"] == "train":
        model_overwrite_config = config["model_collections"][
            config["model"]].pop("overwrite", {})
        config = overwrite_config_from_vargs(config, model_overwrite_config)

    config = overwrite_config_from_vargs(config, vargs)

    if config["mode"] == "train":
        config["model_config"] = config["model_collections"][config["model"]]

    config["dataset_config"] = config["dataset_collections"][config["dataset"]]

    config["vargs"] = vargs

    return config


def overwrite_config_from_vargs(config: dict, vargs: dict):
    for key, value in config.items():
        if isinstance(value, dict):
            overwrite_config_from_vargs(value, vargs)
        elif key in vargs:
            overwrite_config(config, key, vargs[key])
    return config


def unpack_nested_dict(my_dict: dict, new_dict: dict):
    for key, value in my_dict.items():
        if isinstance(value, dict):
            unpack_nested_dict(value, new_dict)
        else:
            new_dict[key] = value


def overwrite_model_config(model, config):
    config.update(model.config)

    unpacked_model_config = {}
    unpack_nested_dict(model.config, unpacked_model_config)
    overwrite_config_from_vargs(config, unpacked_model_config)

    vargs = config.pop("vargs", {})
    config["mode"] = vargs["mode"]
    config["model"] = vargs["model"]
    config["dataset"] = vargs["dataset"]
    config = overwrite_config_from_vargs(config, vargs)
    config["vargs"] = vargs


def eval_node_classification(split,
                             model,
                             loader,
                             enable_tqdm,
                             sampling_strategy,
                             device="cpu",
                             multilabel=False,
                             threshold=0,
                             reverse_mp=False):
    predictions = []
    truths = []
    bar = tqdm(loader, total=len(loader), disable=not enable_tqdm)

    for batch in bar:
        if sampling_strategy == "SAGE":
            mask = torch.arange(batch.batch_size)
        elif sampling_strategy in [None, "None"]:
            mask = eval(f"batch.{split}_mask")
        elif sampling_strategy == "GraphBatching":
            mask = None

        targets = batch.y.to(device)
        outputs = model(**get_batch_input(batch, reverse_mp, device))

        if mask is not None:
            targets = targets[mask]
            outputs = outputs[mask]

        if multilabel:
            preds = outputs > threshold
        else:
            preds = outputs.argmax(dim=-1)

        predictions.append(preds)
        truths.append(targets)

    # Metrics
    predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
    truths = torch.cat(truths, dim=0).detach().cpu().numpy()

    return classification_report(truths, predictions, zero_division=0)


def eval_edge_classification(split,
                             model,
                             loader,
                             enable_tqdm,
                             sampling_strategy,
                             device="cpu",
                             multilabel=False,
                             threshold=0,
                             reverse_mp=False):
    predictions = []
    truths = []
    bar = tqdm(loader, total=len(loader), disable=not enable_tqdm)
    for batch in bar:
        if sampling_strategy == "SAGE":
            # Get edges in batch that are source edges
            batch.src_e_id = loader.data.input_id_to_e_id[batch.input_id]
            mask = torch.isin(batch.e_id, batch.src_e_id)
            in_batch_e_id = batch.e_id[mask]

            # Get source edges that are not in batch
            mask_not_in_batch = ~torch.isin(batch.src_e_id, in_batch_e_id)

            # Append source edges that are not in batch to the batch
            append_source_edges(batch, mask_not_in_batch, loader.data)
            mask = torch.hstack(
                (mask, torch.ones(batch.num_appended, dtype=torch.bool)))

        elif sampling_strategy in [None, "None"]:
            mask = eval(f"batch.{split}_mask")

        targets = batch.y.to(device)
        outputs = model(**get_batch_input(batch, reverse_mp, device))

        if mask is not None:
            targets = targets[mask]
            outputs = outputs[mask]

        if multilabel:
            preds = outputs > threshold
        else:
            preds = outputs.argmax(dim=-1)

        predictions.append(preds)
        truths.append(targets)

    # Metrics
    predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
    truths = torch.cat(truths, dim=0).detach().cpu().numpy()

    return classification_report(truths, predictions, zero_division=0)


def eval_gnn(config):
    vargs = config["vargs"]
    model_name, version = vargs["model"], vargs["version"]

    if version is None:
        client = MlflowClient()
        model_metadata = client.get_registered_model(model_name)
        version = model_metadata.latest_versions[0].version

    dst_path = f"./registered_models/{model_name}/{version}"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    logger.info(f"Model is saved at {dst_path}")
    model = load_pyt_model(model_uri=f'models:/{model_name}/{version}',
                           dst_path=dst_path)

    # Need to overwrite the model configs from the loaded model
    overwrite_model_config(model, config)
    general_config = config["general_config"]
    dataset_config = config["dataset_config"]

    train_loader, val_loader, test_loader = get_loader(config)

    task_type = dataset_config["task_type"]
    if task_type.endswith("NC"):
        eval_step = eval_node_classification
    elif task_type.endswith("EC"):
        eval_step = eval_edge_classification

    reports = {}
    for split, loader in zip(["train", "val", "test"],
                             [train_loader, val_loader, test_loader]):
        reports[split] = eval_step(
            split,
            model,
            loader,
            enable_tqdm=general_config["tqdm"],
            sampling_strategy=general_config["sampling_strategy"],
            device=general_config["device"],
            multilabel=True if task_type.startswith("multi") else False,
            reverse_mp=config["model_config"].get("reverse_mp", False))

    # Save report
    info_message = [""]
    for split, report in reports.items():
        info_message.append(
            f"Evaluation report on the {split} data:\n{report}\n")
    info_message = "\n".join(info_message)
    logger.info(info_message)

    report_path = os.path.join(dst_path,
                               f"Evalution_report_{config['dataset']}.txt")
    logger.info(f"Evaluation reports are saved at {report_path}")
    with open(report_path, "w") as out_file:
        out_file.write(info_message)
