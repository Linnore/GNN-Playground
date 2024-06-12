import os
import argparse
import requests
import json
import mlflow
import torch

from loguru import logger

def create_parser(config):
    parser = argparse.ArgumentParser()

    # Required field
    parser.add_argument('mode', choices=['train', 'inference'])
    parser.add_argument('dataset', choices=list(config["dataset_collections"].keys()))
    parser.add_argument('model', choices=list(config["model_collections"].keys()))
    
    
    # Global settings
    global_settings = parser.add_argument_group("Global Settigns")
    global_settings.add_argument('--seed', type=int, default=None)
    global_settings.add_argument('--device', default=None)
    global_settings.add_argument('--tqdm', action="store_true", default=None)
    global_settings.add_argument('--save_model', action="store_true", default=None)
    global_settings.add_argument('--criterion', type=str, default=None, choices=["loss", "accuracy", "f1"])
    global_settings.add_argument('--patience', type=int, default=None)
    
    
    # Global hyperparameters
    global_params = parser.add_argument_group("Global Hyperparameters")
    inductive_type_group = global_params.add_mutually_exclusive_group()
    inductive_type_group.add_argument('--SAGE', action="store_const", dest='inductive_type', const='SAGE', default='SAGE')
    inductive_type_group.add_argument('--SAINT', action="store_const", dest='inductive_type', const='SAINT')
    global_params.add_argument('--num_epochs', type=int, default=None)
    global_params.add_argument('--batch_size', type=int, default=None)
    global_params.add_argument('--lr', type=float, default=None)
    
    # Model hyperparameters
    model_params = parser.add_argument_group("Model Hyperparameters")
    model_params.add_argument('--num_layers', type=int, default=None)
    model_params.add_argument('--hidden_node_channels', type=int, help='Number of hidden node channels.', default=None)
    model_params.add_argument('--num_neighbors', type=int, nargs="+", default=None)
    model_params.add_argument('--dropout', type=float, default=None)
    model_params.add_argument('--jk', type=str, default=None)
    

    # MLflow settings
    mlflow_group = parser.add_argument_group('MLflow configuration')
    mlflow_group.add_argument('--local_mlflow', action='store_true', default=True)
    mlflow_group.add_argument('--mlflow_server',
                        action='store_false', dest='local_mlflow')
    mlflow_group.add_argument('--auth', action='store_true', help="Indicate whether the remote mlflow tracking server requires authentication. If enable, please provide the credentials in 'mlflow_config.json'.")

    return parser


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_mlflow(config):
    # MLFlow
    mlflow_config = config["mlflow"]
    if config["local_mlflow"]:
        cwd = os.getcwd()
        logger.info(f"MLFlow loggings are under the directory {cwd}")
    else:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_config['tracking_uri']
        if config["auth"]:
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
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        
    # for key, value in vargs.items():
    #     if key in config:
    #         if value!=None and value != config[key]:
    #             logger.warning(f'Overwrite {key}={value} into configuration from argument input.')
    #             config[key] = value
    #     else:
    #         config[key] = value
            
    return config


def update_config(config:dict, vargs:dict):
    for key, value in vargs.items():
        if key in config:
            if value!=None and value != config[key]:
                logger.warning(f'Overwrite {key}={value} into configuration from argument input.')
                config[key] = value
        else:
            config[key] = value


def setup_logger():
    log_dir = "logs"
    tmp_dir = "logs/tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    logger.add(os.path.join(log_dir, "log_{time:YYYY-MM-DD}.txt"), rotation='10 MB')
    
    
