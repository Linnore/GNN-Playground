import os
import yaml
import argparse
import requests
import torch_geometric
import sys
import torch

from loguru import logger
from datetime import datetime


def add_train_parser(subparsers: argparse._SubParsersAction,
                     parent_parser: argparse.ArgumentParser, config: dict):

    parser = subparsers.add_parser("train",
                                   help="Train a model on a dataset.",
                                   parents=[parent_parser])

    # Required Field
    parser.add_argument('--model',
                        choices=list(config["model_collections"].keys()))
    parser.add_argument('--dataset',
                        choices=list(config["dataset_collections"].keys()))

    # Experiment settings
    experiment_config = parser.add_argument_group("Experiment Configurations")
    experiment_config.add_argument('--register_model',
                                   action=argparse.BooleanOptionalAction,
                                   default=None)
    experiment_config.add_argument('--from_run_config', type=str, default=None)

    # System settings
    system_config = parser.add_argument_group("System Configurations")
    system_config.add_argument(
        '--framework',
        '-w',
        choices=config["system_config"]["framework_options"],
        default=None)
    system_config.add_argument('--seed', '-s', type=int, default=None)
    system_config.add_argument('--device', '-d', default=None)
    system_config.add_argument('--tqdm',
                               action=argparse.BooleanOptionalAction,
                               default=None)
    system_config.add_argument('--num_workers', type=int, default=None)
    system_config.add_argument('--verbose',
                               action=argparse.BooleanOptionalAction,
                               default=None)
    system_config.add_argument('--criterion',
                               type=str,
                               default=None,
                               choices=["loss", "accuracy", "f1"])
    system_config.add_argument('--persistent_workers',
                               action=argparse.BooleanOptionalAction,
                               default=None)

    # Training settings
    training_config = parser.add_argument_group("Global Hyperparameters")
    training_config.add_argument('--batch_size', '-b', type=int, default=None)
    training_config.add_argument('--lr', type=float, default=None)
    training_config.add_argument('--weight_decay', type=float, default=None)

    training_config.add_argument('--weighted_BCE',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)
    training_config.add_argument('--CE_weight', type=float, default=None)
    training_config.add_argument('--weighted_CE',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)
    training_config.add_argument('--f1_average', type=str, default=None)
    training_config.add_argument('--num_epochs', type=int, default=None)
    training_config.add_argument('--patience', type=int, default=None)

    # sampling settings
    sampling_config = parser.add_argument_group("Global Sampling settings")
    sampling_config.add_argument(
        '--sampling_strategy',
        choices=config["sampling_config"]["sampling_strategy_options"],
        default=None)
    sampling_config.add_argument('--temporal_sampling',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)
    sampling_config.add_argument('--temporal_strategy', default=None)
    sampling_config.add_argument('--time_attr', default=None)
    sampling_config.add_argument(
        '--SAGE_inductive_option',
        choices=config["sampling_config"]["SAGE_inductive_options"],
        default=None)
    sampling_config.add_argument('--sample_when_predict',
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
                                 choices=[
                                     "single-label-NC",
                                     "multi-label-NC",
                                     "single-label-EC",
                                 ],
                                 default=None)


def add_inference_parser(subparsers: argparse._SubParsersAction,
                         parent_parser: argparse.ArgumentParser, config):
    parser = subparsers.add_parser(
        "inference",
        help="Inference on a dataset's test split using a "
        "registered mlflow model.",
        parents=[parent_parser])

    # Required Field
    parser.add_argument('--dataset',
                        choices=list(config["dataset_collections"].keys()))

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

    # System settings
    system_config = parser.add_argument_group("System Settings")
    system_config.add_argument(
        '--framework',
        choices=config["system_config"]["framework_options"],
        default=None)
    system_config.add_argument('--seed', '-s', type=int, default=None)
    system_config.add_argument('--device', '-d', default=None)
    system_config.add_argument('--tqdm',
                               action=argparse.BooleanOptionalAction,
                               default=None)
    system_config.add_argument('--verbose',
                               action=argparse.BooleanOptionalAction,
                               default=None)

    # sampling settings
    sampling_config = parser.add_argument_group("Global Sampling settings")
    sampling_config.add_argument(
        '--sampling_strategy',
        choices=config["sampling_config"]["sampling_strategy_options"],
        default=None)
    sampling_config.add_argument(
        '--SAGE_inductive_option',
        choices=config["sampling_config"]["SAGE_inductive_options"],
        default=None)
    sampling_config.add_argument('--sample_when_predict',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)
    sampling_config.add_argument('--temporal_sampling',
                                 action=argparse.BooleanOptionalAction,
                                 default=None)


def add_evaluate_parser(subparsers: argparse._SubParsersAction,
                        parent_parser: argparse.ArgumentParser, config):
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a registered model on a dataset",
        parents=[parent_parser])

    # Required Field
    parser.add_argument('--dataset',
                        choices=list(config["dataset_collections"].keys()))

    parser.add_argument(
        '--model', help="Model name of the registerted model to evaluate.")
    parser.add_argument('--version',
                        type=int,
                        default=None,
                        help='Use the latest version if not specified.')

    # System settings
    system_config = parser.add_argument_group("System Configuration")
    system_config.add_argument(
        '--framework',
        choices=config["system_config"]["framework_options"],
        default=None)

    system_config.add_argument('--seed', type=int, default=None)
    system_config.add_argument('--device', default=None)
    system_config.add_argument('--tqdm',
                               action=argparse.BooleanOptionalAction,
                               default=None)

    # Sampling settings
    sampling_config = parser.add_argument_group("Global Sampling settings")
    sampling_config.add_argument(
        '--sampling_strategy',
        choices=config["sampling_config"]["sampling_strategy_options"],
        default=None)
    sampling_config.add_argument(
        '--SAGE_inductive_option',
        choices=config["sampling_config"]["SAGE_inductive_options"],
        default=None)
    sampling_config.add_argument('--sample_when_predict',
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


def create_parser(config: dict):
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
    mlflow_config.add_argument('--mlf_experiment', default=None)
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
            logger.success(f"Successfully logged in to the MLFlow server "
                           f"at {mlflow_config['tracking_uri']} as "
                           f"{mlflow_config['username']}.")

            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_config['username']
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_config['password']

        else:
            raise NotImplementedError("Failed to log in to the MLFlow server "
                                      f"at {mlflow_config['tracking_uri']}!")


def init_config() -> dict:
    curr_dir = os.getcwd()
    config = {}

    config_dir = os.path.join(curr_dir, 'config/pyg')
    for path in os.listdir(config_dir):
        # For config/pyg/*.yaml
        file_name, file_type = os.path.splitext(path)
        if file_type == ".yaml":
            config_name = file_name
            with open(os.path.join(config_dir, path), 'r') as config_file:
                config[config_name] = yaml.safe_load(config_file)

        # For config/pyg/$folder/*.yaml
        elif os.path.isdir(os.path.join(config_dir, path)):
            config[path] = {}  # path is folder name
            config_sub_dir = os.path.join(config_dir, path)
            for sub_path in os.listdir(config_sub_dir):
                file_name, file_type = os.path.splitext(sub_path)
                if file_type == ".yaml":
                    collection_name = file_name
                    with open(os.path.join(config_sub_dir, sub_path),
                              'r') as collection_config:
                        config[path][collection_name] = yaml.safe_load(
                            collection_config)
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


def load_run_config(config: dict):
    exp_config = config["experiment_config"]
    exp_config_file = exp_config["from_run_config"]
    exp_config_file = os.path.join(exp_config["config_dir"], "run",
                                   exp_config_file)
    with open(exp_config_file, 'r') as in_file:
        run_config = yaml.safe_load(in_file)
    return run_config


def unpack_nested_dict(my_dict: dict, new_dict: dict):
    for key, value in my_dict.items():
        if isinstance(value, dict):
            unpack_nested_dict(value, new_dict)
        else:
            new_dict[key] = value


def update_config(config: dict, vargs: dict):
    config["experiment_config"]["config_dir"] = "./config/pyg"
    config["experiment_config"]["from_run_config"] = vargs["from_run_config"]
    config["experiment_config"]["mode"] = vargs["mode"]

    if vargs["from_run_config"]:
        run_config = load_run_config(config)
        model = run_config["experiment_config"]["model"]
        dataset = run_config["experiment_config"]["dataset"]
        assert vargs["model"] is None, (
            "Only support specifying model and dataset"
            "from run_config when run_config is given!!!")
        logger.info("Loading the run configuration")
    else:
        model = vargs["model"]
        dataset = vargs["dataset"]

    config["experiment_config"]["model"] = model
    config["experiment_config"]["dataset"] = dataset

    mode = config["experiment_config"]["mode"]

    if mode == "train":
        model_overwrite_config = config["model_collections"][model].pop(
            "overwrite", {})
        config = overwrite_config_from_vargs(config, model_overwrite_config)

    if vargs["from_run_config"]:
        unpacked_run_config = {}
        unpack_nested_dict(run_config, unpacked_run_config)
        config = overwrite_config_from_vargs(config, unpacked_run_config)

    config = overwrite_config_from_vargs(config, vargs)

    if mode == "train":
        config["model_config"] = config["model_collections"][model]

    config["dataset_config"] = config["dataset_collections"][dataset]

    config["vargs"] = vargs

    return config


def overwrite_config_from_vargs(config: dict, vargs: dict):
    for key, value in config.items():
        if isinstance(value, dict):
            overwrite_config_from_vargs(value, vargs)
        elif key in vargs:
            overwrite_config(config, key, vargs[key])
    return config


def setup_logger(args, config: dict):
    if not args.debug:
        logger.remove(0)
        logger.add(sys.stderr, level="INFO")

    log_dir = "logs"
    tmp_dir = "logs/tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H_%M")
    log_file = os.path.join(log_dir, f"log_{time_str}.log")

    logger.add(log_file, rotation='10 MB')
    config["terminal_log_file"] = log_file
