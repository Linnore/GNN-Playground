import mlflow
import torch

import numpy as np

from tqdm import tqdm
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight


def get_pos_weight_for_BCEWithLogitsLoss(data):
    # TODO: get weights for graph batching
    total_num = data.num_nodes
    pos_cnt = torch.unique(data.y, return_counts=True)[-1]
    neg_cnt = total_num - pos_cnt
    return neg_cnt/pos_cnt


def get_weight_for_CrossEntropyLoss(data):
    # TODO: get weights for graph batching
    y = data.y.numpy()
    weight = compute_class_weight(
        class_weight="balanced", classes=np.unique(y), y=y)
    weight = torch.tensor(weight)


def get_loss_fn(config, loader, reduction="mean"):
    dataset_config = config["dataset_config"]
    if config["general_config"]["sampling_strategy"] != "GraphBatching":
        data = loader.data
    else:
        logger.warning("Weighted loss function is not implemented for graph batching!")
        if config["hyperparameters"]["weighted_CE"] or config["hyperparameters"]["weighted_BCE"]:
            raise NotImplementedError
    
    if dataset_config["task_type"] == "single-label-NC":
        if config["hyperparameters"]["weighted_CE"]:
            weight = get_weight_for_CrossEntropyLoss(data)
        else:
            weight = None
        return torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    elif dataset_config["task_type"] == "multi-label-NC":
        if config["hyperparameters"]["weighted_BCE"]:
            pos_weight = get_pos_weight_for_BCEWithLogitsLoss(data)
        else:
            pos_weight = None
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    elif dataset_config["task_type"] == "single-label-EC":
        if config["hyperparameters"]["weighted_CE"]:
            weight = get_weight_for_CrossEntropyLoss(data)
        else:
            weight = None
        return torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)


def node_classification_step(mode: str, epoch, loader, model, loss_fn, optimizer, enable_tqdm, sampling_strategy, device="cpu", multilabel=False, threshold=0):
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

        targets = batch.y  # on cpu
        outputs = model(batch.x.to(device), batch.edge_index.to(device))

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
        predictions.append(preds)
        truths.append(targets)

        loss = loss.detach().cpu()
        num_targets = outputs.numel()
        total_loss += loss * num_targets
        total_num += num_targets
        bar.set_description(f"{mode}_loss={loss:<8.6g}")

    # Metrics
    predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
    truths = torch.cat(truths, dim=0).detach().numpy()

    avg_loss = total_loss/total_num
    mlflow.log_metric(f"{mode} loss", avg_loss, epoch)

    f1 = f1_score(truths, predictions, average="micro")
    mlflow.log_metric(f"{mode} F1", f1, epoch)

    return avg_loss, f1, predictions, truths


def edge_classification_step(mode: str, epoch, loader, model, loss_fn, optimizer, enable_tqdm, sampling_strategy, device="cpu", multilabel=False, threshold=0):

    total_loss = 0
    total_num = 0
    predictions = []
    truths = []
    has_edge_attr = 'edge_attr' in loader.data.edge_attrs()
    bar = tqdm(loader, total=len(loader), disable=not enable_tqdm)
    for batch in bar:
        if mode == "train":
            optimizer.zero_grad()

        if sampling_strategy == "SAGE":
            mask = torch.isin(batch.e_id, batch.input_id)
        elif sampling_strategy in [None, "None"]:
            mask = eval(f"batch.{mode}_mask")

        targets = batch.y  # on cpu
        outputs = model(batch.x.to(device), batch.edge_index.to(
            device), batch.edge_attr if has_edge_attr else None)

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
        predictions.append(preds)
        truths.append(targets)

        loss = loss.detach().cpu()
        num_targets = outputs.numel()
        total_loss += loss * num_targets
        total_num += num_targets
        bar.set_description(f"{mode}_loss={loss:<8.6g}")

    # Metrics
    predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
    truths = torch.cat(truths, dim=0).detach().numpy()

    avg_loss = total_loss/total_num
    mlflow.log_metric(f"{mode} loss", avg_loss, epoch)

    f1 = f1_score(truths, predictions, average="micro")
    mlflow.log_metric(f"{mode} F1", f1, epoch)

    return avg_loss, f1, predictions, truths


def get_run_step(model, loss_fn, optimizer, sampling_strategy, enable_tqdm, device, task_type):

    if task_type == "single-label-NC":
        run_step = lambda *args, **kwargs: node_classification_step(
            *args,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            enable_tqdm=enable_tqdm,
            sampling_strategy=sampling_strategy,
            device=device,
            **kwargs)
    elif task_type == "multi-label-NC":
        run_step = lambda *args, **kwargs: node_classification_step(
            *args,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            enable_tqdm=enable_tqdm,
            sampling_strategy=sampling_strategy,
            device=device,
            multilabel=True,
            threshold=0,
            **kwargs)
    elif task_type == "single-label-EC":
        run_step = lambda *args, **kwargs: edge_classification_step(
            *args,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            enable_tqdm=enable_tqdm,
            sampling_strategy=sampling_strategy,
            device=device,
            **kwargs
        )
    elif task_type == "single-label-EC":
        run_step = lambda *args, **kwargs: edge_classification_step(
            *args,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            enable_tqdm=enable_tqdm,
            sampling_strategy=sampling_strategy,
            device=device,
            multilabel=True,
            threshold=0,
            **kwargs
        )
    else:
        raise NotImplementedError("Unsupported task type for training.")

    return run_step
