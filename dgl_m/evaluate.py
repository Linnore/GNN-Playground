import torch
import os

from .data_utils import get_loader

from mlflow import MlflowClient
from mlflow.pytorch import load_model as load_pyt_model

from loguru import logger
from tqdm import tqdm
from sklearn.metrics import classification_report


def eval(model, loader, enable_tqdm, sampling_strategy, device="cpu", multilabel=False, threshold=0):
    predictions = []
    truths = []
    bar = tqdm(loader, total=len(loader), disable=not enable_tqdm)

    for graph in bar:
        if sampling_strategy in ["SAGE", None, "None"]:
            # mask = torch.arange(batch_size, device=device)
            mfgs = graph[2]
            targets = mfgs[-1].dstdata['label']
            inputs = mfgs[0].srcdata['feat']
        elif sampling_strategy in ["GraphBatching"]:
            # mask = torch.ones(mfgs[0].srcdata['feat'].shape[0], dtype=bool)
            mfgs = graph
            targets = mfgs.ndata['label']
            inputs = mfgs.ndata['feat']
            
        # targets = batch.y[mask] # on cpu
        # outputs = model(batch.x.to(device), batch.edge_index.to(device))[mask]
        outputs = model(mfgs, inputs)
        
        if multilabel:
            preds = outputs > threshold
        else:
            preds = outputs.argmax(dim=-1)
            
        predictions.append(preds)
        truths.append(targets)
        
    # Metrics
    predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
    truths = torch.cat(truths, dim=0).detach().numpy()
    
    return classification_report(
        truths, predictions, zero_division=0
    )
    

def overwrite_model_config(model, config):
    config.update(model.config)


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
    model = load_pyt_model(
        model_uri=f'models:/{model_name}/{version}',
        dst_path=dst_path
    )

    # Need to overwrite the model configs from the loaded model
    overwrite_model_config(model, config)
    general_config = config["general_config"]
    dataset_config = config["dataset_config"]

    train_loader, val_loader, test_loader = get_loader(config)

    reports = {}
    for split, loader in zip(["train", "val", "test"], [train_loader, val_loader, test_loader]):
        reports[split] = eval(
            model,
            loader,
            enable_tqdm=general_config["tqdm"],
            sampling_strategy=general_config["sampling_strategy"],
            device=general_config["device"],
            multilabel= True if dataset_config["task_type"].startswith("multi") else False
        )
    
    # Save report
    info_message = [""]
    for split, report in reports.items():
        info_message.append(f"Evaluation report on the {split} data:\n{report}\n")
    info_message = "\n".join(info_message)
    logger.info(info_message)

    report_path = os.path.join(dst_path, f"Evalution_report_{config['dataset']}.txt")
    logger.info(f"Evaluation reports are saved at {report_path}")
    with open(report_path, "w") as out_file:
        out_file.write(info_message)
        
        