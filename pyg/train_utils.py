import mlflow
import torch

import numpy as np

from tqdm import tqdm
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from mlflow.types.schema import Schema, TensorSpec


def get_batch_input(batch, reverse_mp, device):
    input_dict = {}
    input_dict["x"] = batch.x.to(device)
    input_dict["edge_index"] = batch.edge_index.to(device)
    if reverse_mp:
        input_dict["rev_edge_index"] = input_dict["edge_index"].flip(0)
    if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
        input_dict["edge_attr"] = batch.edge_attr.to(device)
        if reverse_mp:
            input_dict["rev_edge_attr"] = batch.rev_edge_attr.to(device)

    return input_dict


def append_source_edges(batch, mask_not_in_batch, data):
    batch.edge_index = torch.hstack(
        (batch.edge_index, batch.edge_label_index[:, mask_not_in_batch]))
    batch.y = torch.hstack((batch.y, batch.edge_label[mask_not_in_batch]))

    # Retrieve edge attributes from the hole data object
    missing_e_id = batch.src_e_id[mask_not_in_batch]
    if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
        batch.edge_attr = torch.vstack(
            (batch.edge_attr, data.edge_attr[missing_e_id]))
        if hasattr(batch, "rev_edge_attr"):
            batch.rev_edge_attr = torch.vstack(
                (batch.rev_edge_attr, data.rev_edge_attr[missing_e_id]))
    batch.num_appended = missing_e_id.shape[0]


def get_io_schema(sample_input: dict, dataset_config: dict, reverse_mp):
    input_list = [
        TensorSpec(np.dtype(np.float32),
                   (-1, dataset_config["num_node_features"]), "x"),
        TensorSpec(np.dtype(np.int64), (2, -1), "edge_index"),
    ]
    if "edge_attr" in sample_input:
        input_list.append(
            TensorSpec(np.dtype(np.float32),
                       (-1, sample_input["edge_attr"].shape[1]), "edge_attr"))
    if "rev_edge_index" in sample_input:
        input_list.append(
            TensorSpec(np.dtype(np.float32), (2, -1), "rev_edge_index"))
    if "rev_edge_attr" in sample_input:
        input_list.append(
            TensorSpec(np.dtype(np.float32),
                       (-1, sample_input["rev_edge_attr"].shape[1]),
                       "rev_edge_attr"))

    input_schema = Schema(input_list)
    output_schema = Schema([
        TensorSpec(np.dtype(np.float32), (-1, dataset_config["num_classes"]))
    ])

    return input_schema, output_schema


def get_pos_weight_for_BCEWithLogitsLoss(data):
    # TODO: get weights for graph batching
    total_num = data.num_nodes
    pos_cnt = torch.unique(data.y, return_counts=True)[-1]
    neg_cnt = total_num - pos_cnt
    pos_weight = neg_cnt / pos_cnt
    logger.info(f"Loss weight: pos_weight={pos_weight}")
    return pos_weight


def get_weight_for_CrossEntropyLoss(data, config):
    weight = config["hyperparameters"].get("CE_weight", "auto")
    if weight == "auto":
        # TODO: get weights for graph batching
        y = data.y.numpy()
        weight = compute_class_weight(class_weight="balanced",
                                      classes=np.unique(y),
                                      y=y)
    weight = torch.tensor(weight, dtype=torch.float32)
    logger.info(f"Loss weight: weight={weight}")
    return weight


def get_loss_fn(config, loader, reduction="mean"):
    dataset_config = config["dataset_config"]
    if config["general_config"]["sampling_strategy"] != "GraphBatching":
        data = loader.data
    else:
        logger.warning(
            "Weighted loss function is not implemented for graph batching!")
        if config["hyperparameters"]["weighted_CE"] or config[
                "hyperparameters"]["weighted_BCE"]:
            raise NotImplementedError

    if dataset_config["task_type"] == "single-label-NC":
        if config["hyperparameters"]["weighted_CE"]:
            weight = get_weight_for_CrossEntropyLoss(data, config)
        else:
            weight = None
        return torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    elif dataset_config["task_type"] == "multi-label-NC":
        if config["hyperparameters"]["weighted_BCE"]:
            pos_weight = get_pos_weight_for_BCEWithLogitsLoss(data)
        else:
            pos_weight = None
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight,
                                          reduction=reduction)

    elif dataset_config["task_type"] == "single-label-EC":
        if config["hyperparameters"]["weighted_CE"]:
            weight = get_weight_for_CrossEntropyLoss(data, config)
        else:
            weight = None
        return torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)


def infer_licit_x(edge_index, edge_label):
    pass


def node_metrics_for_node_readout():
    pass


def node_metrics_for_edge_readout(truth_e, prediction_e, f1_average="binary"):

    pass


def edge_metrics_for_edge_readout():
    pass


def node_classification_step(mode: str,
                             epoch,
                             loader,
                             model,
                             loss_fn,
                             optimizer,
                             enable_tqdm,
                             sampling_strategy,
                             device="cpu",
                             multilabel=False,
                             threshold=0,
                             reverse_mp=False,
                             f1_average="micro"):
    total_loss = 0
    total_num = 0
    predictions = []
    truths = []
    bar = tqdm(loader, total=len(loader), disable=not enable_tqdm)
    for batch in bar:
        if mode == "train":
            optimizer.zero_grad()

        if sampling_strategy == "SAGE":
            mask = torch.arange(batch.batch_size)
        elif sampling_strategy in [None, "None"]:
            mask = eval(f"batch.{mode}_mask")
        elif sampling_strategy == "GraphBatching":
            mask = None

        targets = batch.y.to(device).long()
        outputs = model(**get_batch_input(batch, reverse_mp, device))

        if mask is not None:
            targets = targets[mask]
            outputs = outputs[mask]

        loss = loss_fn(outputs, targets)

        if mode == "train":
            loss.backward()
            optimizer.step()

        if multilabel:
            preds = outputs > threshold
        else:
            preds = outputs.argmax(dim=-1)
        predictions.append(preds.detach().cpu().numpy())
        truths.append(targets.detach().cpu().numpy())

        loss = loss.detach().cpu().item()
        num_targets = outputs.numel()
        total_loss += loss * num_targets
        total_num += num_targets
        bar.set_description(f"{mode}_loss={loss:<8.6g}")

    # Metrics
    predictions = np.concatenate(predictions)
    truths = np.concatenate(truths)

    avg_loss = total_loss / total_num
    mlflow.log_metric(f"{mode} loss", avg_loss, epoch)

    f1 = f1_score(truths, predictions, average=f1_average)
    mlflow.log_metric(f"{mode} F1", f1, epoch)

    return avg_loss, f1, predictions, truths


def edge_classification_step(mode: str,
                             epoch,
                             loader,
                             model,
                             loss_fn,
                             optimizer,
                             enable_tqdm,
                             sampling_strategy,
                             device="cpu",
                             multilabel=False,
                             threshold=0,
                             reverse_mp=False,
                             f1_average="binary",
                             use_node_metrics=False):

    total_loss = 0
    total_num = 0
    predictions = []
    truths = []
    bar = tqdm(loader, total=len(loader), disable=not enable_tqdm)
    for batch in bar:
        if mode == "train":
            optimizer.zero_grad()

        if sampling_strategy == "SAGE":
            # Get edges in batch that are source edges
            batch.src_e_id = batch.input_id_to_e_id(batch.input_id)
            mask = torch.isin(batch.e_id, batch.src_e_id)
            in_batch_e_id = batch.e_id[mask]

            # Get source edges that are not in batch
            mask_not_in_batch = ~torch.isin(batch.src_e_id, in_batch_e_id)

            # Append source edges that are not in batch to the batch
            append_source_edges(batch, mask_not_in_batch, loader.data)
            mask = torch.hstack(
                (mask, torch.ones(batch.num_appended, dtype=torch.bool)))

        elif sampling_strategy in [None, "None"]:
            mask = eval(f"batch.{mode}_mask")
        elif sampling_strategy == "GraphBatching":
            mask = None

        targets = batch.y.to(device).long()
        outputs = model(**get_batch_input(batch, reverse_mp, device))

        if mask is not None:
            targets = targets[mask]
            outputs = outputs[mask]

        loss = loss_fn(outputs, targets)

        if mode == "train":
            loss.backward()
            optimizer.step()

        if multilabel:
            preds = outputs > threshold
        else:
            preds = outputs.argmax(dim=-1)
        predictions.append(preds.detach().cpu().numpy())
        truths.append(targets.detach().cpu().numpy())

        loss = loss.detach().cpu().item()
        num_targets = outputs.numel()
        total_loss += loss * num_targets
        total_num += num_targets
        bar.set_description(f"{mode}_loss={loss:<8.6g}")

    # Metrics
    predictions = np.concatenate(predictions)
    truths = np.concatenate(truths)

    avg_loss = total_loss / total_num
    mlflow.log_metric(f"{mode} loss", avg_loss, epoch)

    # Note that 1 is the minority class
    if use_node_metrics:
        f1 = node_metrics_for_edge_readout(truths,
                                           predictions,
                                           average=f1_average)
    else:
        f1 = edge_metrics_for_edge_readout()
        f1 = f1_score(truths, predictions, average=f1_average)
    mlflow.log_metric(f"{mode} F1", f1, epoch)

    return avg_loss, f1, predictions, truths


def single_label_NC_step(model, loss_fn, optimizer, sampling_strategy,
                         enable_tqdm, device, reverse_mp, f1_average):
    return lambda *args, **kwargs: node_classification_step(
        *args,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        enable_tqdm=enable_tqdm,
        sampling_strategy=sampling_strategy,
        device=device,
        reverse_mp=reverse_mp,
        f1_average=f1_average,
        **kwargs)


def multi_label_NC_step(model, loss_fn, optimizer, sampling_strategy,
                        enable_tqdm, device, reverse_mp, f1_average):
    return lambda *args, **kwargs: node_classification_step(
        *args,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        enable_tqdm=enable_tqdm,
        sampling_strategy=sampling_strategy,
        device=device,
        reverse_mp=reverse_mp,
        f1_average=f1_average,
        multilabel=True,
        threshold=0,
        **kwargs)


def single_label_EC_step(model, loss_fn, optimizer, sampling_strategy,
                         enable_tqdm, device, reverse_mp, f1_average):
    return lambda *args, **kwargs: edge_classification_step(
        *args,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        enable_tqdm=enable_tqdm,
        sampling_strategy=sampling_strategy,
        device=device,
        reverse_mp=reverse_mp,
        f1_average=f1_average,
        **kwargs)


def multi_label_EC_step(model, loss_fn, optimizer, sampling_strategy,
                        enable_tqdm, device, reverse_mp, f1_average):
    return lambda *args, **kwargs: edge_classification_step(
        *args,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        enable_tqdm=enable_tqdm,
        sampling_strategy=sampling_strategy,
        device=device,
        reverse_mp=reverse_mp,
        f1_average=f1_average,
        multilabel=True,
        threshold=0,
        **kwargs)


def get_run_step(**kwargs):
    task_type = kwargs.pop('task_type')
    if task_type == "single-label-NC":
        run_step = single_label_NC_step(**kwargs)
    elif task_type == "multi-label-NC":
        run_step = multi_label_NC_step(**kwargs)
    elif task_type == "single-label-EC":
        run_step = single_label_EC_step(**kwargs)
    elif task_type == "single-label-EC":
        run_step = multi_label_EC_step(**kwargs)
    else:
        raise NotImplementedError("Unsupported task type for training.")

    return run_step
