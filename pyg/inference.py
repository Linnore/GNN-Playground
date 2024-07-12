import torch
import os

import numpy as np
import pandas as pd

from .data_utils import get_inference_loader

from mlflow import MlflowClient
from mlflow.pytorch import load_model as load_pyt_model

from loguru import logger
from tqdm import tqdm


def node_classification_inference(model, loader, split, enable_tqdm, sampling_strategy, device="cpu", multilabel=False, threshold=0):
    n_ids = []
    predictions = []
    bar = tqdm(loader, total=len(loader), disable=not enable_tqdm)

    for batch in bar:
        if sampling_strategy == "SAGE":
            mask = torch.arange(batch.batch_size)
            batch_n_ids = batch.n_id[mask]
        elif sampling_strategy in [None, "None"]:
            mask = eval(f"batch.{split}_mask")
            batch_n_ids = torch.arange(batch.num_nodes)[mask]
        elif sampling_strategy == "GraphBatching":
            mask = torch.ones(batch.x.shape[0], dtype=bool)
            raise NotImplementedError

        outputs = model(batch.x.to(device), batch.edge_index.to(device))[mask]

        if multilabel:
            preds = outputs > threshold
        else:
            preds = outputs.argmax(dim=-1)

        n_ids.append(batch_n_ids)
        predictions.append(preds)

    # Metrics
    n_ids = torch.cat(n_ids, dim=0).detach().numpy()
    predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()

    idx = np.argsort(n_ids)
    n_ids = n_ids[idx]
    predictions = predictions[idx]

    return n_ids, predictions


def overwrite_model_config(model, config):
    config.update(model.config)


def infer_gnn(config):
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
    model = load_pyt_model(
        model_uri=f'models:/{model_name}/{version}',
        dst_path=dst_path
    )

    # Need to overwrite the model configs from the loaded model
    overwrite_model_config(model, config)
    general_config = config["general_config"]
    dataset_config = config["dataset_config"]

    loader = get_inference_loader(config)
    data = loader.data

    n_ids, predictions = node_classification_inference(
        model,
        loader,
        split=vargs['split'],
        enable_tqdm=vargs["tqdm"],
        sampling_strategy=general_config["sampling_strategy"],
        device=general_config["device"],
        multilabel=True if dataset_config["task_type"].startswith(
            "multi") else False
    )

    # Save predictions
    pred_path = os.path.join(
        vargs["output_dir"],
        f"{config['model']}-v{version}-{config['dataset']}",
        f"{vargs['split']}-pred"
    )+".csv"

    save_dir = os.path.split(pred_path)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n = n_ids.shape[0]
    content = np.hstack((n_ids.reshape(n, -1), predictions.reshape(n, -1)))
    pred_df = pd.DataFrame(content)
    col_names = ["n_id"]
    if predictions.ndim > 1:
        for i in range(predictions.shape[1]):
            col_names.append(f"pred_{i+1}")
    else:
        col_names.append("pred")

    pred_df.columns = col_names
    pred_df.to_csv(pred_path, index=False)

    truth_path = os.path.join(save_dir, f"{vargs['split']}-truth.csv")

    content = np.hstack(
        (n_ids.reshape(n, -1), data.y.numpy()[n_ids].reshape(n, -1)))
    truth_df = pd.DataFrame(content)
    truth_df.columns = col_names
    truth_df.to_csv(truth_path, index=False)
