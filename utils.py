import os
import argparse
import requests
import mlflow
import torch_geometric
import config
import sys
import torch

from loguru import logger


def add_train_parser(subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser, config: config):

    parser = subparsers.add_parser(
        "train", help="Train a model on a dataset.", parents=[parent_parser])

    # General settings
    general_config = parser.add_argument_group("Global Settigns")
    general_config.add_argument(
        '--framework', choices=config.framework_options, default=None)
    general_config.add_argument(
        '--sampling_strategy', choices=config.sampling_strategy_options, default=None)
    general_config.add_argument(
        '--SAGE_inductive_option', choices=config.SAGE_inductive_options, default=None)
    general_config.add_argument(
        '--sample_when_predict', action=argparse.BooleanOptionalAction, default=None)
    general_config.add_argument('--seed', type=int, default=None)
    general_config.add_argument('--device', default=None)
    general_config.add_argument('--tqdm', action="store_true", default=None)
    general_config.add_argument(
        '--save_model', action="store_true", default=None)
    general_config.add_argument(
        '--criterion', type=str, default=None, choices=["loss", "accuracy", "f1"])
    general_config.add_argument('--num_epochs', type=int, default=None)
    general_config.add_argument('--patience', type=int, default=None)
    general_config.add_argument('--num_workers', type=int, default=None)

    # General hyperparameters
    hyperparameters = parser.add_argument_group("Global Hyperparameters")
    hyperparameters.add_argument('--batch_size', type=int, default=None)
    hyperparameters.add_argument('--lr', type=float, default=None)
    hyperparameters.add_argument('--weight_decay', type=float, default=None)

    # Model hyperparameters
    model_params = parser.add_argument_group("Model Hyperparameters")
    model_params.add_argument('--hidden_node_channels_per_head', type=int,
                              help='Number of hidden node channels per head.', default=None)
    model_params.add_argument('--num_layers', type=int, default=None)
    model_params.add_argument('--heads', type=int, default=None)
    model_params.add_argument('--output_heads', type=int, default=None)
    model_params.add_argument(
        '--num_neighbors', type=int, nargs="+", default=None)
    model_params.add_argument('--dropout', type=float, default=None)
    model_params.add_argument('--jk', type=str, default=None)
    model_params.add_argument('--v2', action="store_true", default=None)


def add_inference_parser(subparsers, parent_parser, config):
    parser = subparsers.add_parser(
        "inference", help="Inference a model on a dataset's test split.", parents=[parent_parser])


def create_parser(config: config):
    parser = argparse.ArgumentParser(
        "Run GNN experiment implemented using Pytorch Geometric.")

    # Parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--debug', action="store_true")

    # Required field
    parent_parser.add_argument(
        'model', choices=list(config.model_collections.keys()))
    parent_parser.add_argument('dataset', choices=list(
        config.dataset_collections.keys()))

    # MLflow settings
    mlflow_config = parent_parser.add_argument_group('MLflow configuration')
    mlflow_config.add_argument('--tracking_uri', default=None)
    mlflow_config.add_argument('--username', default=None)
    mlflow_config.add_argument('--password', default=None)
    mlflow_config.add_argument('--experiment', default=None)
    mlflow_config.add_argument('--auth', action=argparse.BooleanOptionalAction,
                               help="Indicate whether the remote mlflow tracking server requires authentication. If enable, please provide the credentials in 'mlflow_config.json'.", default=None)

    # Create subparsers
    subparsers = parser.add_subparsers(dest="mode", required=True)
    add_train_parser(subparsers, parent_parser, config)
    add_inference_parser(subparsers, parent_parser, config)

    args = parser.parse_args()
    return args


def set_global_seed(seed):
    torch_geometric.seed_everything(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
            logger.exception(
                f"Failed to log in to the MLFlow server at {mlflow_config['tracking_uri']}!")

    logger.info(f"Launching experiment: {mlflow_config['experiment']}")
    mlflow.set_experiment(mlflow_config["experiment"])


def init_config():
    from config import config
    return config


OVERWRITE_MESSAGE_SET = set()


def overwrite_config(config, key, value):
    global OVERWRITE_MESSAGE_SET
    if value != None:
        if value != config[key]:
            message = f'Overwrite {key}={value} from {key}={config[key]}'
            if not message in OVERWRITE_MESSAGE_SET:
                OVERWRITE_MESSAGE_SET.add(message)
                logger.warning(
                    f'Overwrite {key}={value} from {key}={config[key]}')
        config[key] = value


def update_config(config: dict, vargs: dict):
    config["model"] = vargs["model"]
    config["mode"] = vargs["mode"]
    config["dataset"] = vargs["dataset"]

    model_overwrite_config = config["model_collections"][config["model"]].pop(
        "overwrite", {})
    config = overwrite_config_from_vargs(config, model_overwrite_config)

    config = overwrite_config_from_vargs(config, vargs)
    return config


def overwrite_config_from_vargs(config: dict, vargs: dict):
    for key, value in config.items():
        if type(value) == dict:
            overwrite_config_from_vargs(value, vargs)
        elif key in vargs:
            overwrite_config(config, key, vargs[key])
    return config


def setup_logger(args):
    if not args.debug:
        logger.remove(0)
        logger.add(sys.stderr, level="INFO")

    log_dir = "logs"
    tmp_dir = "logs/tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    logger.add(os.path.join(
        log_dir, "log_{time:YYYY-MM-DD-HH_mm}.txt"), rotation='10 MB')
