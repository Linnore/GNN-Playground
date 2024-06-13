import os
import argparse
import requests
import json
import mlflow
import torch_geometric
import config

from loguru import logger

def create_parser(config:config):
    parser = argparse.ArgumentParser()

    # Required field
    parser.add_argument('mode', choices=['train', 'inference'])
    parser.add_argument('model', choices=list(config.model_collections.keys()))
    parser.add_argument('dataset', choices=list(config.dataset_collections.keys()))
    
    # MLflow settings
    mlflow_config = parser.add_argument_group('MLflow configuration')
    mlflow_config.add_argument('--tracking_uri',default=None)
    mlflow_config.add_argument('--username',default=None)
    mlflow_config.add_argument('--password',default=None)
    mlflow_config.add_argument('--experiment',default=None)
    mlflow_config.add_argument('--auth', action='store_true', help="Indicate whether the remote mlflow tracking server requires authentication. If enable, please provide the credentials in 'mlflow_config.json'.", default=None)
    
    # General settings
    general_config = parser.add_argument_group("Global Settigns")
    general_config.add_argument('--framework', choices=config.framework_options, default=None)
    general_config.add_argument('--seed', type=int, default=None)
    general_config.add_argument('--device', default=None)
    general_config.add_argument('--tqdm', action="store_true", default=None)
    general_config.add_argument('--save_model', action="store_true", default=None)
    general_config.add_argument('--criterion', type=str, default=None, choices=["loss", "accuracy", "f1"])
    general_config.add_argument('--patience', type=int, default=None)
    general_config.add_argument('--num_workers', type=int, default=None)
    
    
    # General hyperparameters
    hyperparameters = parser.add_argument_group("Global Hyperparameters")
    hyperparameters.add_argument('--sampling_strategy', choices=config.sampling_strategy_options, default=None)
    hyperparameters.add_argument('--SAGE_option', choices=config.SAGE_options, default=None)
    hyperparameters.add_argument('--num_epochs', type=int, default=None)
    hyperparameters.add_argument('--batch_size', type=int, default=None)
    hyperparameters.add_argument('--lr', type=float, default=None)
    
    # Model hyperparameters
    model_params = parser.add_argument_group("Model Hyperparameters")
    model_params.add_argument('--num_layers', type=int, default=None)
    model_params.add_argument('--hidden_node_channels', type=int, help='Number of hidden node channels.', default=None)
    model_params.add_argument('--num_neighbors', type=int, nargs="+", default=None)
    model_params.add_argument('--dropout', type=float, default=None)
    model_params.add_argument('--jk', type=str, default=None)
    model_params.add_argument('--v2', action="store_true", default=None)
    

    return parser


def set_global_seed(seed):
    torch_geometric.seed_everything(seed)


def setup_mlflow(config):
    # MLFlow
    mlflow_config = config["mlflow_config"]
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_config['tracking_uri']
    if mlflow_config["auth"]:
        response = requests.get(
            f"{mlflow_config['tracking_uri']}",
            auth=(mlflow_config["username"], mlflow_config["password"])
        )
        if (response.status_code == 200):
            logger.success(
                f"Successfully logged in to the MLFlow server at {mlflow_config['tracking_uri']} as {mlflow_config['username']}.")

            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_config['username']
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_config['password']
            
        else:
            logger.exception(f"Failed to log in to the MLFlow server at {mlflow_config['tracking_uri']}!")
            
    logger.info(f"Launching experiment: {mlflow_config['experiment']}")
    mlflow.set_experiment(mlflow_config["experiment"])
            
    
def init_config():
    # with open("config.json", "r") as config_file:
    #     config = json.load(config_file)
    from config import config
    return config


def overwrite_config(config, key, value):
    if value != None:
        if value != config[key]:
            logger.warning(f'Overwrite {key}={value} from {key}={config[key]}')
        config[key] = value


def update_config(config:dict, vargs:dict):
    for key, value in config.items():
        if type(value)==dict:
            update_config(value, vargs)
        elif key in vargs:
            overwrite_config(config, key, vargs[key])
    return config
        


def setup_logger():
    log_dir = "logs"
    tmp_dir = "logs/tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    logger.add(os.path.join(log_dir, "log_{time:YYYY-MM-DD-HH_mm}.txt"), rotation='10 MB')
    
    
