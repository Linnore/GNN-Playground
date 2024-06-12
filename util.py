import os
import argparse
import requests
import json
import mlflow
import torch

from loguru import logger

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', choices=['train', 'inference'])
    parser.add_argument('dataset', choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('model', choices=['GraphSAGE', 'GAT', 'GIN'])
    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_neighbors', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--hidden_node_channels', type=int, help='Number of hidden node channels.', default=64)
    parser.add_argument('--criterion', type=str, default="loss", choices=["loss", "accuracy", "f1"])
    parser.add_argument('--save_model', action="store_true")
    parser.add_argument('--save_dir', default="model")
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--tqdm', action="store_true")
    
    # SAGE or SAINT
    inductive_type_group = parser.add_mutually_exclusive_group()
    inductive_type_group.add_argument('--SAGE', action="store_const", dest='inductive_type', const='SAGE', default='SAGE')
    inductive_type_group.add_argument('--SAINT', action="store_const", dest='inductive_type', const='SAINT')


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
            
    
def setup_config(vargs:dict):
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        
    for key, value in vargs.items():
        if key in config:
            if value!=None and value != config[key]:
                logger.warning(f'Overwrite {key}={value} into configuration from argument input.')
                config[key] = value
        else:
            config[key] = value
            
    return config


def setup_logger():
    log_dir = "logs"
    tmp_dir = "logs/tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    logger.add(os.path.join(log_dir, "log_{time:YYYY-MM-DD}.txt"), rotation='10 MB')
    
    
