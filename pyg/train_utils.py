import mlflow
import torch

import numpy as np

from tqdm import tqdm
from loguru import logger
from sklearn.metrics import (f1_score, roc_auc_score)
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
            if hasattr(batch, "rev_edge_attr"):
                input_dict["rev_edge_attr"] = batch.rev_edge_attr.to(device)
            else:
                input_dict["rev_edge_attr"] = batch.edge_attr.to(device)

    return input_dict


def append_source_edges(batch, mask_not_in_batch, data, temporal, time_attr):
    batch.edge_index = torch.hstack(
        (batch.edge_index, batch.edge_label_index[:, mask_not_in_batch]))
    batch.y = torch.hstack((batch.y, batch.edge_label[mask_not_in_batch]))

    # Retrieve edge attributes from the whole data object
    missing_e_id = batch.src_e_id[mask_not_in_batch]
    if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
        batch.edge_attr = torch.vstack(
            (batch.edge_attr, data.edge_attr[missing_e_id]))
        if hasattr(batch, "rev_edge_attr"):
            batch.rev_edge_attr = torch.vstack(
                (batch.rev_edge_attr, data.rev_edge_attr[missing_e_id]))
    # Retrieve edge timestamps from the whole data object
    if temporal:
        batch[time_attr] = torch.hstack(
            (batch[time_attr], data[time_attr][missing_e_id]))
    batch.num_appended = missing_e_id.shape[0]


def append_source_nodes_by_self_loops(batch, temporal, time_attr):
    batch.edge_index = torch.hstack((batch.edge_index, batch.edge_label_index))
    batch.y = torch.hstack((batch.y, batch.edge_label))
    batch.num_appended = batch.edge_label_index.shape[1]
    if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
        batch.edge_attr = torch.vstack(
            (batch.edge_attr,
             torch.zeros((batch.num_appended, batch.edge_attr.shape[1]))))
        if hasattr(batch, "rev_edge_attr"):
            batch.rev_edge_attr = torch.vstack(
                (batch.rev_edge_attr,
                 torch.zeros((batch.num_appended, batch.edge_attr.shape[1]))))
    if temporal:
        batch[time_attr] = torch.hstack(
            (batch[time_attr], batch.edge_label_time))


def get_io_schema(sample_input: dict, dataset_config: dict):
    input_list = [
        TensorSpec(np.dtype(np.float32),
                   ("num_nodes", dataset_config["num_node_features"]), "x"),
        TensorSpec(np.dtype(np.int64), (2, "num_edges"), "edge_index"),
    ]
    if "edge_attr" in sample_input:
        input_list.append(
            TensorSpec(np.dtype(np.float32),
                       ("num_edges", sample_input["edge_attr"].shape[1]),
                       "edge_attr"))
    if "rev_edge_index" in sample_input:
        input_list.append(
            TensorSpec(np.dtype(np.float32), (2, "num_edges"),
                       "rev_edge_index"))
    if "rev_edge_attr" in sample_input:
        input_list.append(
            TensorSpec(np.dtype(np.float32),
                       ("num_edges", sample_input["rev_edge_attr"].shape[1]),
                       "rev_edge_attr"))

    input_schema = Schema(input_list)

    if dataset_config["task_type"] in ["single-label-NC", "multi-label-NC"]:
        output_first_dim = "num_nodes"
    elif dataset_config["task_type"] in ["single-label-EC", "multi-label-EC"]:
        output_first_dim = "num_edges"
    elif dataset_config["task_type"] in ["single-label-dynamic_NC"]:
        output_first_dim = "num_node_time_query"
    else:
        None
    output_schema = Schema([
        TensorSpec(np.dtype(np.float32),
                   (output_first_dim, dataset_config["num_classes"]))
    ])

    return input_schema, output_schema


def get_pos_weight_for_BCEWithLogitsLoss(data, config):
    weight = config["training_config"].get("BCE_weight", "auto")
    if weight is None or weight == "auto":
        # TODO: get weights for graph batching
        total_num = data.num_nodes
        pos_cnt = torch.unique(data.y, return_counts=True)[-1]
        neg_cnt = total_num - pos_cnt
        pos_weight = neg_cnt / pos_cnt
        logger.info(f"Loss weight: pos_weight={pos_weight}")
        return pos_weight
    weight = torch.tensor(weight, dtype=torch.float32)
    logger.info(f"Loss weight: weight={weight}")
    return weight


def get_weight_for_CrossEntropyLoss(data, config):
    weight = config["training_config"].get("CE_weight", "auto")
    if weight is None or weight == "auto":
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
    sampling_config = config["sampling_config"]
    training_config = config["training_config"]
    loss_fn = training_config["loss_fn"]
    task_type = dataset_config["task_type"]

    if loss_fn is None:
        if task_type in ["single-label-NC", "single-label-EC"]:
            loss_fn = "CE"
        elif task_type in ["multi-label-NC", "multi-label-EC"]:
            loss_fn = "BCE"
        elif task_type in ["single-label-dynamic_NC"]:
            loss_fn = "Focal"
        else:
            raise NotImplementedError
        training_config["loss_fn"] = loss_fn
        logger.warning("loss_fn is not set! Automatically select "
                       f"loss_fn={loss_fn} for a {task_type} task!")

    if loss_fn == "Focal":
        alpha = training_config["focal_alpha"]
        gamma = training_config["focal_gamma"]
        logger.info(f"Using Focal Loss with alpha={alpha}, gamma={gamma}")

        if task_type in [
                "single-label-dynamic_NC", "single-label-NC", "single-label-EC"
        ]:
            if dataset_config["num_classes"] > 2:
                raise NotImplementedError(
                    "Focal loss only supports binary classification.")
            elif dataset_config["num_classes"] == 2:
                dataset_config["num_classes"] = 1
            else:
                raise ValueError("num_classes should be >= 2.")

            from torchvision.ops.focal_loss import sigmoid_focal_loss

            class FocalLoss(torch.nn.Module):

                def __init__(self, gamma=0, alpha=None, reduction="mean"):
                    super(FocalLoss, self).__init__()
                    self.gamma = gamma
                    self.alpha = alpha
                    self.reduction = reduction

                def forward(self, input, target):
                    return sigmoid_focal_loss(input.flatten(), target.float(),
                                              self.alpha, self.gamma,
                                              self.reduction)

            return FocalLoss(gamma, alpha, reduction=reduction)
        else:
            raise NotImplementedError(
                f"Focal loss is not implemented for {task_type}")

    elif loss_fn == "CE":
        if sampling_config["sampling_strategy"] != "GraphBatching":
            data = loader.data
        else:
            if (training_config["weighted_CE"] and training_config["CE_weight"]
                    is None) or (training_config["weighted_BCE"]):
                raise NotImplementedError(
                    "Auto-weights for weighted CE/BCE is not "
                    "implemented for graph batching!")
            data = None

        if task_type in [
                "single-label-NC", "single-label-EC", "single-label-dynamic_NC"
        ]:
            if training_config["weighted_CE"]:
                weight = get_weight_for_CrossEntropyLoss(data, config)
            else:
                weight = None
            return torch.nn.CrossEntropyLoss(weight=weight,
                                             reduction=reduction)
        else:
            raise NotImplementedError(
                f"CrossEntropy loss is not implemented for {task_type}")

    elif loss_fn == "BCE":
        if sampling_config["sampling_strategy"] != "GraphBatching":
            data = loader.data
        else:
            if (training_config["weighted_CE"] and training_config["CE_weight"]
                    is None) or (training_config["weighted_BCE"]):
                raise NotImplementedError(
                    "Auto-weights for weighted CE/BCE is not "
                    "implemented for graph batching!")
            data = None
        if task_type in ["multi-label-NC"]:
            if training_config["weighted_BCE"]:
                pos_weight = get_pos_weight_for_BCEWithLogitsLoss(data, config)
            else:
                pos_weight = None
            return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight,
                                              reduction=reduction)
        elif task_type in [
                "single-label-NC", "single-label-EC", "single-label-dynamic_NC"
        ]:
            if dataset_config["num_classes"] > 2:
                raise NotImplementedError(
                    f"BCELoss for {task_type} only supports `num_classes`=2.")
            elif dataset_config["num_classes"] == 2:
                dataset_config["num_classes"] = 1
            else:
                raise ValueError("num_classes should be >= 2.")

            if training_config["weighted_BCE"]:
                pos_weight = get_pos_weight_for_BCEWithLogitsLoss(data, config)
            else:
                pos_weight = None
            return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight,
                                              reduction=reduction)
        else:
            raise NotImplementedError(
                f"BinaryCrossEntropy loss is not implemented for {task_type}")
    else:
        raise NotImplementedError("Unknown loss function")


def node_classification_step(mode: str,
                             epoch,
                             loader,
                             model,
                             loss_fn,
                             optimizer,
                             enable_tqdm,
                             sampling_strategy,
                             device="cpu",
                             use_threshold=False,
                             reverse_mp=False,
                             **kwargs):
    total_loss = 0
    total_num = 0
    truths = []
    prob_scores = []
    bar = tqdm(loader, total=len(loader), disable=not enable_tqdm)
    for batch in bar:
        if mode == "train":
            optimizer.zero_grad()

        if sampling_strategy == "SAGE":
            mask = torch.arange(batch.batch_size)
        elif sampling_strategy in [None, "None"]:
            mask = batch[f"{mode}_mask"]
        elif sampling_strategy == "GraphBatching":
            mask = None

        if use_threshold:
            targets = batch.y.to(device).float()
        else:
            targets = batch.y.to(device).long()
        outputs = model(**get_batch_input(batch, reverse_mp, device))

        if mask is not None:
            targets = targets[mask]
            outputs = outputs[mask]

        loss = loss_fn(outputs, targets)

        if mode == "train":
            loss.backward()
            optimizer.step()

        if use_threshold:
            scores = torch.sigmoid(outputs)
        else:
            scores = torch.softmax(outputs, dim=-1)

        truths.append(targets.detach().cpu().numpy())
        prob_scores.append(scores.detach().cpu().numpy())

        loss = loss.detach().cpu().item()
        num_targets = outputs.numel()
        total_loss += loss * num_targets
        total_num += num_targets
        bar.set_description(f"{mode}_loss={loss:<8.6g}")

    results = {}
    truths = np.concatenate(truths)
    prob_scores = np.concatenate(prob_scores)

    avg_loss = total_loss / total_num
    mlflow.log_metric(f"{mode} loss", avg_loss, epoch)

    results["loss"] = avg_loss
    results["prob_scores"] = prob_scores
    results["truths"] = truths
    results["predictions"] = compute_predictions(use_threshold, prob_scores)
    return results


def edge_classification_step(
    mode: str,
    epoch,
    loader,
    model,
    loss_fn,
    optimizer,
    enable_tqdm,
    sampling_strategy,
    device="cpu",
    temporal_sampling=False,
    time_attr="time",
    use_threshold=False,
    reverse_mp=False,
    **kwargs,
):

    total_loss = 0
    total_num = 0
    truths = []
    prob_scores = []
    bar = tqdm(loader, total=len(loader), disable=not enable_tqdm)
    for batch in bar:
        if mode == "train":
            optimizer.zero_grad()

        if sampling_strategy == "SAGE":
            # Get edges in batch that are source edges
            batch.src_e_id = loader.data.input_id_to_e_id[batch.input_id]
            mask = torch.isin(batch.e_id, batch.src_e_id)
            in_batch_e_id = batch.e_id[mask]

            # Get source edges that are not in batch
            mask_not_in_batch = ~torch.isin(batch.src_e_id, in_batch_e_id)

            # Append source edges that are not in batch to the batch
            append_source_edges(batch, mask_not_in_batch, loader.data,
                                temporal_sampling, time_attr)
            mask = torch.hstack(
                (mask, torch.ones(batch.num_appended, dtype=torch.bool)))

        elif sampling_strategy in [None, "None"]:
            mask = batch[f"{mode}_mask"]
        elif sampling_strategy == "GraphBatching":
            mask = None

        if use_threshold:
            targets = batch.y.to(device).float()
        else:
            targets = batch.y.to(device).long()
        outputs = model(**get_batch_input(batch, reverse_mp, device))

        if mask is not None:
            targets = targets[mask]
            outputs = outputs[mask]

        loss = loss_fn(outputs, targets)

        if mode == "train":
            loss.backward()
            optimizer.step()

        if use_threshold:
            scores = torch.sigmoid(outputs)
        else:
            scores = torch.softmax(outputs, dim=-1)

        truths.append(targets.detach().cpu().numpy())
        prob_scores.append(scores.detach().cpu().numpy())

        loss = loss.detach().cpu().item()
        num_targets = outputs.numel()
        total_loss += loss * num_targets
        total_num += num_targets
        bar.set_description(f"{mode}_loss={loss:<8.6g}")

    results = {}
    truths = np.concatenate(truths)
    prob_scores = np.concatenate(prob_scores)

    avg_loss = total_loss / total_num
    mlflow.log_metric(f"{mode} loss", avg_loss, epoch)

    results["loss"] = avg_loss
    results["prob_scores"] = prob_scores
    results["truths"] = truths
    results["predictions"] = compute_predictions(use_threshold, prob_scores)
    return results


def dynamic_node_classification_step(
    mode: str,
    epoch,
    loader,
    model,
    loss_fn,
    optimizer,
    enable_tqdm,
    sampling_strategy,
    device="cpu",
    use_threshold=False,
    reverse_mp=False,
    **kwargs,
):
    total_loss = 0
    total_num = 0
    truths = []
    prob_scores = []
    bar = tqdm(loader, total=len(loader), disable=not enable_tqdm)
    for batch in bar:
        if mode == "train":
            optimizer.zero_grad()

        if sampling_strategy == "SAGE":
            targets = batch.edge_label.long()
            target_nodes = batch.edge_label_index[0, :]
        elif sampling_strategy == "GraphBatching":
            # This is for snapshot node classification
            targets = batch.y.long()
            target_nodes = batch.target_nodes + batch.ptr[:batch.batch_size]
        else:
            NotImplementedError

        embedding = model.encode(
            **get_batch_input(batch, reverse_mp, device))[target_nodes, :]

        outputs = model.decode(embedding)
        loss = loss_fn(outputs, targets)

        if mode == "train":
            loss.backward()
            optimizer.step()

        if use_threshold:
            scores = torch.sigmoid(outputs)
        else:
            scores = torch.softmax(outputs, dim=-1)

        truths.append(targets.detach().cpu().numpy())
        prob_scores.append(scores.detach().cpu().numpy())

        loss = loss.detach().cpu().item()
        num_targets = outputs.numel()
        total_loss += loss * num_targets
        total_num += num_targets
        bar.set_description(f"{mode}_loss={loss:<8.6g}")

    results = {}
    truths = np.concatenate(truths)
    prob_scores = np.concatenate(prob_scores)

    avg_loss = total_loss / total_num
    mlflow.log_metric(f"{mode} loss", avg_loss, epoch)

    results["loss"] = avg_loss
    results["prob_scores"] = prob_scores
    results["truths"] = truths
    results["predictions"] = compute_predictions(use_threshold, prob_scores)
    return results


def compute_predictions(
    use_threshold,
    prob_scores,
    threshold=0.5,
):
    if use_threshold:
        predictions = prob_scores > threshold
    else:
        predictions = prob_scores.argmax(axis=-1)
    return predictions


def compute_metrics(
    mode,
    epoch,
    prob_scores,
    truths,
    predictions,
    compute_f1=False,
    compute_auc=False,
    f1_average="binary",
    auc_average="macro",
):
    metrics = {}
    if compute_f1:
        f1 = f1_score(truths, predictions, average=f1_average)
        metrics["f1"] = f1
        mlflow.log_metric(f"{mode} F1", f1, epoch)

    if compute_auc:
        auc = roc_auc_score(truths,
                            prob_scores,
                            average=auc_average,
                            multi_class="ovo")
        metrics["auc"] = auc
        mlflow.log_metric(f"{mode} AUC", auc, epoch)

    return metrics


def get_run_step(task_type, run_step_kwargs):
    if run_step_kwargs["loss_fn_name"] in ["Focal", "BCE"]:
        run_step_kwargs["use_threshold"] = True
    elif run_step_kwargs["loss_fn_name"] in ["CE"]:
        run_step_kwargs["use_threshold"] = False
    if task_type == "single-label-NC":
        return node_classification_step, run_step_kwargs
    elif task_type == "multi-label-NC":
        return node_classification_step, run_step_kwargs
    elif task_type == "single-label-EC":
        return edge_classification_step, run_step_kwargs
    elif task_type == "multi-label-EC":
        return edge_classification_step, run_step_kwargs
    elif task_type == "single-label-dynamic_NC":
        return dynamic_node_classification_step, run_step_kwargs
    else:
        raise NotImplementedError("Unsupported task type for training.")
