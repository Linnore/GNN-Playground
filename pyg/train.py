import torch
import mlflow
import pprint
import os

from .data_utils import get_loader
from .model.GraphSAGE import GraphSAGE_PyG
from .model.GAT import GAT_PyG, GAT_Custom

from loguru import logger
from torch_geometric.nn import summary
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score

def get_model(config):
    model_config = config["model_collections"][config["model"]]
    dataset_config = config["dataset_collections"][config["dataset"]]
    match model_config["base_model"]:
        case "GraphSAGE_PyG":
            model = GraphSAGE_PyG(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config["hidden_node_channels"],
                num_layers=model_config["num_layers"],
                dropout=model_config["dropout"],
                jk=model_config["jk"],
            )
        case "GAT_PyG":
            model = GAT_PyG(
                in_channels=dataset_config["num_node_features"],
                out_channels=dataset_config["num_classes"],
                hidden_channels=model_config["hidden_node_channels"],
                num_layers=model_config["num_layers"],
                dropout=model_config["dropout"],
                jk=model_config["jk"],
            )
        case "GAT_Custom":
            model = GAT_Custom(
                
            )
            
        case "GIN_PyG":
            pass
        case _:
            logger.exception(f"Unreconized base model: {model_config['base_model']}")
    
    return model


def train_gnn(config):
    
    general_config = config["general_config"]    
    device = general_config["device"]
    
    # Initialize MLflow Logging
    run_name = f"{config['model']}-{config['dataset']}"
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tag("base model", config["model_collections"][config["model"]]["base_model"])
    mlflow.set_tag("dataset", config["dataset"])
    logger.info(f"Launching run: {run.info.run_name}")
    
    
    # Log hyperparameters
    params = config["hyperparameters"]
    params.update(config["model_collections"][config["model"]])
    params_str = pprint.pformat(params)
    logger.info(f"General configurations:\n{general_config}")
    logger.info(f"Hyperparameters:\n{params_str}")
    mlflow.log_params(general_config)
    mlflow.log_params(params)
    
    # Get loaders
    train_loader, val_loader, test_loader = get_loader(config)
    
    # Get model
    model = get_model(config).to(device)
    
    # Setup save directory for optimizer states
    if general_config["save_model"]:
        save_path = os.path.join("logs/tmp", f"{run.info.run_name}-Optimizer-{run.info.run_id}.tar")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
            
    # Summary logging
    sample_batch=next(iter(train_loader)).to(device)
    sample_x = sample_batch.x
    sample_edge_index = sample_batch.edge_index
    summary_str = summary(model, sample_x, sample_edge_index)
    logger.info("Model Summary:\n" + summary_str)
    with open("logs/tmp/model_summary.txt", "w") as out_file:
        out_file.write(summary_str)
    mlflow.log_artifact("logs/tmp/model_summary.txt")
    
    # Setup Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    
    # Setup loss function
    logger.warning('Todo: Implement WCE.')
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean') 
    
    # Setup metrics
    criterion = general_config["criterion"].lower()
    if criterion=="loss":
        best_value = -2147483647
    elif criterion in ["accuracy", "f1"]:
        best_value = -1
    
    # Training loop
    patience = general_config["patience"]
    if patience == None:
        patience = general_config["num_epochs"]
    train_steps = 0
    best_epoch = 0
    for epoch in range(1, 1+general_config["num_epochs"]):
        if general_config["tqdm"]:
            print(f"Epoch {epoch}:")
        # Batch training
        model.train()
        total_train_loss = num_train = 0
        predictions = []
        truths = []
        train_bar = tqdm(train_loader, total=len(train_loader), disable=not general_config["tqdm"])
        for batch in train_bar:
            optimizer.zero_grad()
            batch = batch.to(device)
            targets = batch.y[:batch.batch_size]
            outputs = model(batch.x, batch.edge_index)[:batch.batch_size]
            
            predictions.append(outputs.argmax(dim=-1))
            truths.append(targets)
            
            loss = loss_fn(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            num_batch_train = outputs.numel()
            loss = loss.detach().cpu() 
            total_train_loss += loss * num_batch_train
            num_train += num_batch_train
            
            train_steps += 1
            train_bar.set_description(f"Train_loss={loss:<8.6g}")
            mlflow.log_metric("Running Loss", loss, step=train_steps)
        
        # Metrics on training data
        train_loss = total_train_loss/num_train
        predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
        truths = torch.cat(truths, dim=0).detach().cpu().numpy()
        train_acc = accuracy_score(truths, predictions)
        train_f1 = f1_score(truths, predictions, average="weighted")
        
        mlflow.log_metric("Train Loss", train_loss, epoch)
        mlflow.log_metric("Train Accuracy", train_acc, epoch)
        mlflow.log_metric("Train F1", train_f1, epoch)
        
        
        # Validation
        model.eval()
        total_val_loss = num_val = 0
        predictions = []
        truths = []
        val_bar = tqdm(val_loader, total=len(val_loader), disable=not general_config["tqdm"])
        for batch in val_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            
            
            targets = batch.y[:batch.batch_size]
            outputs = model(batch.x, batch.edge_index)[:batch.batch_size]
            
            predictions.append(outputs.argmax(dim=-1))
            truths.append(targets)
            
            loss = loss_fn(outputs, targets)
            
            num_batch_val = outputs.numel()
            loss = loss.detach().cpu()
            total_val_loss += loss * num_batch_val
            num_val += num_batch_val
            
            val_bar.set_description(f"  Val_loss={loss:<8.6g}")
        
        # Metrics on validation data
        val_loss = total_val_loss/num_val
        predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
        truths = torch.cat(truths, dim=0).detach().cpu().numpy()
        val_acc = accuracy_score(truths, predictions)
        val_f1 = f1_score(truths, predictions, average="weighted")
        
        mlflow.log_metric("Validation Loss", total_val_loss/num_val, epoch)
        mlflow.log_metric("Validation Accuracy", val_acc, epoch)
        mlflow.log_metric("Validation F1", val_f1, epoch)
        
        
        # Test
        num_test = 0
        predictions = []
        truths = []
        test_bar = tqdm(test_loader, desc=f"Testing", total=len(test_loader), disable=not general_config["tqdm"])
        for batch in test_bar:
            optimizer.zero_grad()
            batch = batch.to(device)
            targets = batch.y[:batch.batch_size]
            outputs = model(batch.x, batch.edge_index)[:batch.batch_size]
            
            predictions.append(outputs.argmax(dim=-1))
            truths.append(targets)
            
            num_test += outputs.numel()

        
        # Metrics on testing data
        predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
        truths = torch.cat(truths, dim=0).detach().cpu().numpy()
        test_acc = accuracy_score(truths, predictions)
        test_f1 = f1_score(truths, predictions, average="weighted")
        mlflow.log_metric("Testing Accuracy", test_acc, epoch)
        
        logger.info(f"Epoch {epoch}: train_loss={train_loss:<8.6g}, train_acc={train_acc:<8.6g}, val_loss={val_loss:<8.6g}, val_acc={val_acc:<8.6g}, test_acc={test_acc:<8.6g}, test_f1={test_f1:<8.6g}")
        
        # Best model
        if criterion=="loss":
            criterion_value = -val_loss
        elif criterion=="accuracy":
            criterion_value = val_acc
        elif criterion=="f1":
            criterion_value = val_f1
            
        if criterion_value > best_value:
            best_value = criterion_value
            mlflow.log_metric("Best Test Accuracy", test_acc, epoch)
            mlflow.log_metric("Best Test F1", test_f1, epoch)
            
            best_model_state_dict = model.state_dict()
            best_report = classification_report(truths, predictions, zero_division=0)
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
        if epoch-best_epoch>patience:
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

