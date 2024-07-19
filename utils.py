import os
import argparse
import requests
import torch_geometric
import config
import sys
import torch

from loguru import logger


def add_train_parser(subparsers: argparse._SubParsersAction,
                     parent_parser: argparse.ArgumentParser, config: config):

    parser = subparsers.add_parser("train",
                                   help="Train a model on a dataset.",
                                   parents=[parent_parser])

    # Required Field
    parser.add_argument('model', choices=list(config.model_collections.keys()))
    parser.add_argument('dataset',
                        choices=list(config.dataset_collections.keys()))

    # General settings
    general_config = parser.add_argument_group("Global Settigns")
    general_config.add_argument('--framework',
                                choices=config.framework_options,
                                default=None)
    general_config.add_argument('--sampling_strategy',
                                choices=config.sampling_strategy_options,
                                default=None)
    general_config.add_argument('--SAGE_inductive_option',
                                choices=config.SAGE_inductive_options,
                                default=None)
    general_config.add_argument('--sample_when_predict',
                                action=argparse.BooleanOptionalAction,
                                default=None)
    general_config.add_argument('--seed', type=int, default=None)
    general_config.add_argument('--device', default=None)
    general_config.add_argument('--tqdm',
                                action=argparse.BooleanOptionalAction,
                                default=None)
    general_config.add_argument('--register_model',
                                action=argparse.BooleanOptionalAction,
                                default=None)
    general_config.add_argument('--criterion',
                                type=str,
                                default=None,
                                choices=["loss", "accuracy", "f1"])
    general_config.add_argument('--num_epochs', type=int, default=None)
    general_config.add_argument('--patience', type=int, default=None)
    general_config.add_argument('--num_workers', type=int, default=None)

    # General hyperparameters
    hyperparameters = parser.add_argument_group("Global Hyperparameters")
    hyperparameters.add_argument('--batch_size', type=int, default=None)
    hyperparameters.add_argument('--lr', type=float, default=None)
    hyperparameters.add_argument('--weight_decay', type=float, default=None)

    # Loss function hyperparameters
    hyperparameters.add_argument('--weighted_BCE',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)
    hyperparameters.add_argument('--weighted_CE',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)

    # Model hyperparameters
    model_params = parser.add_argument_group("Model Hyperparameters")
    model_params.add_argument('--hidden_channels_per_head',
                              type=int,
                              help='Number of hidden node channels per head.',
                              default=None)
    model_params.add_argument('--num_layers', type=int, default=None)
    model_params.add_argument('--heads', type=int, default=None)
    model_params.add_argument('--output_heads', type=int, default=None)
    model_params.add_argument('--num_neighbors',
                              type=int,
                              nargs="+",
                              default=None)
    model_params.add_argument('--dropout', type=float, default=None)
    model_params.add_argument('--jk', type=str, default=None)
    model_params.add_argument('--v2',
                              action=argparse.BooleanOptionalAction,
                              default=None)
    model_params.add_argument('--edge_update',
                              action=argparse.BooleanOptionalAction,
                              default=None)
    model_params.add_argument('--batch_norm',
                              action=argparse.BooleanOptionalAction,
                              default=None)
    model_params.add_argument('--reverse_mp',
                              action=argparse.BooleanOptionalAction,
                              default=None)
    model_params.add_argument('--layer_mix',
                              choices=["None", "Mean", "Sum", "Max", "Cat"])
    model_params.add_argument('--model_mix', choices=["Mean", "Sum", "Max"])

    # AMLworld configuration
    AMLworld_config = parser.add_argument_group("Arguments for AMLworld.")
    AMLworld_config.add_argument('--add_time_stamp',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)
    AMLworld_config.add_argument('--add_egoID',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)
    AMLworld_config.add_argument('--add_port',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)
    AMLworld_config.add_argument('--add_time_delta',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)
    AMLworld_config.add_argument('--ibm_split',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)
    AMLworld_config.add_argument('--force_reload',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)
    AMLworld_config.add_argument('--task_type',
                                 choices=config.task_type_options,
                                 default=None)
    general_config.add_argument('--verbose',
                                action=argparse.BooleanOptionalAction,
                                default=None)


def add_inference_parser(subparsers: argparse._SubParsersAction,
                         parent_parser: argparse.ArgumentParser,
                         config: config):
    parser = subparsers.add_parser(
        "inference",
        help="Inference on a dataset's test split using a "
        "registered mlflow model.",
        parents=[parent_parser])

    # Required Field
    parser.add_argument('dataset',
                        choices=list(config.dataset_collections.keys()))

    parser.add_argument(
        '--model', help="Model name of the registerted model to evaluate.")

    parser.add_argument('--version',
                        type=int,
                        default=None,
                        help='Use the latest version if not specified.')

    parser.add_argument(
        '--split',
        choices=["train", "val", "test", "unlabelled"],
        default="test",
        help='Select the split of the data set to predict. Supported values '
        '[train, val, test] for labelled split, and [unlabelled] for '
        'unlabelled split. The dataset should be able to loaded as a pytorch '
        'geometric dataset, and each data object has the attribute '
        '{split_name}_mask to get the split for inference.')
    parser.add_argument('--output_dir', default="./output")

    # General settings
    general_config = parser.add_argument_group("Global Settigns")
    general_config.add_argument('--framework',
                                choices=config.framework_options,
                                default=None)
    general_config.add_argument('--sampling_strategy',
                                choices=config.sampling_strategy_options,
                                default=None)
    general_config.add_argument('--SAGE_inductive_option',
                                choices=config.SAGE_inductive_options,
                                default=None)
    general_config.add_argument('--sample_when_predict',
                                action=argparse.BooleanOptionalAction,
                                default=None)
    general_config.add_argument('--seed', type=int, default=None)
    general_config.add_argument('--device', default=None)
    general_config.add_argument('--tqdm',
                                action=argparse.BooleanOptionalAction,
                                default=None)
    general_config.add_argument('--verbose',
                                action=argparse.BooleanOptionalAction,
                                default=None)


def add_evaluate_parser(subparsers: argparse._SubParsersAction,
                        parent_parser: argparse.ArgumentParser,
                        config: config):
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a registered model on a dataset",
        parents=[parent_parser])

    # Required Field
    parser.add_argument('dataset',
                        choices=list(config.dataset_collections.keys()))

    parser.add_argument(
        '--model', help="Model name of the registerted model to evaluate.")
    parser.add_argument('--version',
                        type=int,
                        default=None,
                        help='Use the latest version if not specified.')

    # General settings
    general_config = parser.add_argument_group("Global Settigns")
    general_config.add_argument('--framework',
                                choices=config.framework_options,
                                default=None)
    general_config.add_argument('--sampling_strategy',
                                choices=config.sampling_strategy_options,
                                default=None)
    general_config.add_argument('--SAGE_inductive_option',
                                choices=config.SAGE_inductive_options,
                                default=None)
    general_config.add_argument('--sample_when_predict',
                                action=argparse.BooleanOptionalAction,
                                default=None)
    general_config.add_argument('--seed', type=int, default=None)
    general_config.add_argument('--device', default=None)
    general_config.add_argument('--tqdm',
                                action=argparse.BooleanOptionalAction,
                                default=None)

    # AMLworld configuration.
    # Allow evaluate on a different splitting setting to the training one
    AMLworld_config = parser.add_argument_group("Arguments for AMLworld.")
    AMLworld_config.add_argument('--ibm_split',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)
    AMLworld_config.add_argument('--force_reload',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)


def create_parser(config: config):
    parser = argparse.ArgumentParser(
        "Run GNN experiments implemented using Pytorch Geometric.")

    # Parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--debug',
                               action=argparse.BooleanOptionalAction)

    # MLflow settings
    mlflow_config = parent_parser.add_argument_group('MLflow configuration')
    mlflow_config.add_argument('--tracking_uri', default=None)
    mlflow_config.add_argument('--username', default=None)
    mlflow_config.add_argument('--password', default=None)
    mlflow_config.add_argument('--experiment', default=None)
    mlflow_config.add_argument(
        '--auth',
        action=argparse.BooleanOptionalAction,
        help="Indicate whether the remote mlflow tracking server requires "
        "authentication. If enable, please provide the credentials in "
        "'mlflow_config.json'.",
        default=None)

    # Create subparsers
    subparsers = parser.add_subparsers(dest="mode", required=True)
    add_train_parser(subparsers, parent_parser, config)
    add_evaluate_parser(subparsers, parent_parser, config)
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
        response = requests.get(f"{mlflow_config['tracking_uri']}",
                                auth=(mlflow_config["username"],
                                      mlflow_config["password"]))
        if (response.status_code == 200):
            logger.success(
                f"Successfully logged in to the MLFlow server "
                f"at {mlflow_config['tracking_uri']} as "
                f"{mlflow_config['username']}."
            )

            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_config['username']
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_config['password']

        else:
            raise NotImplementedError(
                "Failed to log in to the MLFlow server "
                f"at {mlflow_config['tracking_uri']}!"
            )


def init_config():
    from config import config
    return config


OVERWRITE_MESSAGE_SET = set()


def overwrite_config(config, key, value):
    global OVERWRITE_MESSAGE_SET
    if value is not None:
        if value != config[key]:
            message = f'Overwrite {key}={value} from {key}={config[key]}'
            if message not in OVERWRITE_MESSAGE_SET:
                OVERWRITE_MESSAGE_SET.add(message)
                logger.warning(
                    f'Overwrite {key}={value} from {key}={config[key]}')
        config[key] = value


def update_config(config: dict, vargs: dict):
    config["mode"] = vargs["mode"]
    config["model"] = vargs["model"]
    config["dataset"] = vargs["dataset"]

    if config["mode"] == "train":
        model_overwrite_config = config["model_collections"][
            config["model"]].pop("overwrite", {})
        config = overwrite_config_from_vargs(config, model_overwrite_config)

    config = overwrite_config_from_vargs(config, vargs)

    if config["mode"] == "train":
        config["model_config"] = config["model_collections"][config["model"]]

    config["dataset_config"] = config["dataset_collections"][config["dataset"]]

    config["vargs"] = vargs

    return config


def overwrite_config_from_vargs(config: dict, vargs: dict):
    for key, value in config.items():
        if isinstance(value, dict):
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

    logger.add(os.path.join(log_dir, "log_{time:YYYY-MM-DD-HH_mm}.txt"),
               rotation='10 MB')
