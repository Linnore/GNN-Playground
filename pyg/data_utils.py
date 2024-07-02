import torch
import torch_geometric.transforms as T

from torch_geometric.data import Data, Batch
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, DataLoader

from copy import deepcopy
from loguru import logger


def merge_from_data_list(data_list):
    batch_data = Batch.from_data_list(data_list)
    data = Data(
        x=batch_data.x,
        edge_index=batch_data.edge_index,
        y=batch_data.y
    )
    # TODO: also support edge features and edge lable.

    return data


def get_data_SAGE(config):
    transform = T.Compose([T.NormalizeFeatures()])
    dataset = config["dataset"]
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid('dataset', dataset,
                            split='public', transform=transform)
    elif dataset == "Reddit":
        from torch_geometric.datasets import Reddit
        dataset = Reddit('dataset/Reddit', transform=transform)
    elif dataset == "Reddit2":
        from torch_geometric.datasets import Reddit2
        dataset = Reddit2('dataset/Reddit2', transform=transform)
    elif dataset == "Flickr":
        from torch_geometric.datasets import Flickr
        dataset = Flickr('dataset/Flickr', transform=transform)
    elif dataset == " Yelp":
        from torch_geometric.datasets import Yelp
        dataset = Yelp('dataset/Yelp', transform=transform)
    elif dataset == "AmazonProducts":
        from torch_geometric.datasets import AmazonProducts
        dataset = AmazonProducts('dataset/AmazonProducts', transform=transform)
    elif dataset == "PPI":
        from torch_geometric.datasets import PPI
        dataset = [
            merge_from_data_list(PPI('dataset/PPI', split='train')),
            merge_from_data_list(PPI('dataset/PPI', split='val')),
            merge_from_data_list(PPI('dataset/PPI', split='test')),
        ]
    elif dataset == "AMLworld-HI-Small":
        from pyg.custom_dataset.AMLworld import AMLworld, AddEgoIds
        logger.warning("AddEgoIDs not implemented!!!")
        dataset = AMLworld('dataset/AMLworld', opt="HI-Small")
    else:
        logger.exception('Unsupported dataset.')

    general_config = config["general_config"]

    # Node Classification
    task_type = config["dataset_config"]["task_type"]
    if task_type in ["single-label-NC", "multi-label-NC"]:
        # For dataset containing one graph
        if len(dataset) == 1:
            data = dataset[0]
            if general_config["framework"] == "transductive":
                logger.info("Using data split for transductive training.")
                train_data = data
                val_data = data
                test_data = data

            elif general_config["framework"] == "inductive":
                if general_config["SAGE_inductive_option"] in ["default", "strict"]:
                    logger.info(
                        "Using data split for strict inductive learning.")
                    train_data = data.subgraph(data.train_mask)
                    val_data = data.subgraph(data.val_mask)
                    test_data = data.subgraph(data.test_mask)
                elif general_config["SAGE_inductive_option"] == "soft":
                    logger.info(
                        "Using data split for non-strict inductive learning.")
                    train_data = data.subgraph(data.train_mask)
                    val_data = data
                    test_data = data
        # For dataset containing multiple graphs. Strictly inductive learning by default.
        else:
            train_data, val_data, test_data = dataset

            train_data.train_mask = torch.ones(
                train_data.num_nodes, dtype=bool)
            val_data.val_mask = torch.ones(val_data.num_nodes, dtype=bool)
            test_data.test_mask = torch.ones(test_data.num_nodes, dtype=bool)

    elif task_type in ["single-label-EC"]:
        if len(dataset) == 1:
            data = dataset[0]
            if general_config["framework"] == "transductive":
                logger.info("Using data split for transductive training.")
                train_data = data
                val_data = data
                test_data = data

            elif general_config["framework"] == "inductive":
                if general_config["SAGE_inductive_option"] in ["default", "strict"]:
                    logger.info(
                        "Using data split for strict inductive learning.")
                    train_data = data.edge_subgraph(data.train_mask)
                    val_data = data.edge_subgraph(data.val_mask)
                    test_data = data.edge_subgraph(data.test_mask)

                elif general_config["SAGE_inductive_option"] == "soft":
                    logger.info(
                        "Using data split for non-strict inductive learning.")
                    train_data = data.edge_subgraph(data.train_mask)
                    val_data = data
                    test_data = data
        # For dataset containing multiple graphs. Strictly inductive learning by default.
        else:
            train_data, val_data, test_data = dataset

            train_data.train_mask = torch.ones(
                train_data.num_edges, dtype=bool)
            val_data.val_mask = torch.ones(val_data.num_edges, dtype=bool)
            test_data.test_mask = torch.ones(test_data.num_edges, dtype=bool)

    else:
        logger.exception("Unsupported task type!")

    return train_data, val_data, test_data


def get_data_SAINT(config):
    dataset_config = config["dataset_config"]
    logger.exception('Not Implemented.')


def get_data_graph_batch(config):
    if config["dataset"] == "PPI":
        from torch_geometric.datasets import PPI
        train_dataset = PPI('dataset/PPI', split='train')
        val_dataset = PPI('dataset/PPI', split='val')
        test_dataset = PPI('dataset/PPI', split='test')

    return train_dataset, val_dataset, test_dataset


def get_loader_SAGE(train_data, val_data, test_data, config):
    model_config = config["model_config"]
    params = config["hyperparameters"]

    num_neighbors = model_config.get("num_neighbors", -1)
    if type(num_neighbors) == int:
        num_neighbors = [num_neighbors] * model_config["num_layers"]
    elif type(num_neighbors) == list:
        num_neighbors = num_neighbors
        assert len(num_neighbors) == model_config["num_layers"]

    general_config = config["general_config"]

    logger.info(
        f"\ntrain_data={train_data}\nval_data={val_data}\ntest_data={test_data}")

    task_type = config["dataset_config"]["task_type"]
    if task_type in ["single-label-NC", "multi-label-NC"]:
        train_loader = NeighborLoader(
            train_data,
            num_neighbors=num_neighbors.copy(),
            batch_size=params["batch_size"],
            input_nodes=train_data.train_mask,
            num_workers=general_config["num_workers"],
            persistent_workers=general_config["persistent_workers"],
        )

        if not general_config["sample_when_predict"]:
            logger.warning(
                "sample_when_predict is set to be False. All neighbors will be used for aggregation when doing prediction in validation and testing.")
            num_neighbors = [-1] * model_config["num_layers"]

        val_loader = NeighborLoader(
            val_data,
            num_neighbors=num_neighbors,
            batch_size=params["batch_size"],
            input_nodes=val_data.val_mask,
            num_workers=general_config["num_workers"],
            persistent_workers=general_config["persistent_workers"],
        )

        test_loader = NeighborLoader(
            test_data,
            num_neighbors=num_neighbors,
            batch_size=params["batch_size"],
            input_nodes=test_data.test_mask,
            num_workers=general_config["num_workers"],
            persistent_workers=general_config["persistent_workers"],
        )

    elif task_type in ["single-label-EC"]:
        train_loader = LinkNeighborLoader(
            train_data,
            num_neighbors=num_neighbors,
            batch_size=params["batch_size"],
            edge_label_index=train_data.edge_index[:, train_data.train_mask],
            edge_label=train_data.y
        )

        val_loader = LinkNeighborLoader(
            val_data,
            num_neighbors=num_neighbors,
            batch_size=params["batch_size"],
            edge_label_index=val_data.edge_index[:, val_data.val_mask],
            edge_label=val_data.y
        )

        test_loader = LinkNeighborLoader(
            test_data,
            num_neighbors=num_neighbors,
            batch_size=params["batch_size"],
            edge_label_index=test_data.edge_index[:, test_data.test_mask],
            edge_label=test_data.y
        )

    return train_loader, val_loader, test_loader


def get_loader_SAINT(data: Data, config):
    logger.exception("Not implemented.")


def get_loader_no_sampling(train_data, val_data, test_data, config):
    logger.warning(
        "Sampling strategy is set to be None. Full graph will be used without mini-batching! Batch_size is ignored! ")

    logger.info(
        f"\ntrain_data={train_data}\nval_data={val_data}\ntest_data={test_data}")
    
    train_loader = DataLoader([train_data])
    val_loader = DataLoader([val_data])
    test_loader = DataLoader([test_data])

    return train_loader, val_loader, test_loader


def get_loader_graph_batch(train_dataset, val_dataset, test_dataset, config):
    batch_size = config["hyperparameters"]["batch_size"]
    general_config = config["general_config"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=general_config["num_workers"],
        persistent_workers=general_config["persistent_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=general_config["num_workers"],
        persistent_workers=general_config["persistent_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=general_config["num_workers"],
        persistent_workers=general_config["persistent_workers"],
    )

    return train_loader, val_loader, test_loader


def get_data(config):
    if config["general_config"]["sampling_strategy"] == 'SAGE':
        return get_data_SAGE(config)
    elif config["general_config"]["sampling_strategy"] == 'SAINT':
        return get_data_SAINT(config)


def get_loader(config):
    sampling_strategy = config["general_config"]["sampling_strategy"]
    if sampling_strategy == 'SAGE':
        return get_loader_SAGE(*get_data_SAGE(config), config)
    elif sampling_strategy == 'SAINT':
        return get_loader_SAINT(*get_data_SAINT(config), config)
    elif sampling_strategy == 'GraphBatching':
        return get_loader_graph_batch(*get_data_graph_batch(config), config)
    elif sampling_strategy == 'None' or sampling_strategy == None:
        return get_loader_no_sampling(*get_data_SAGE(config), config)


def get_inference_data_SAGE(config):
    dataset = config["dataset"]
    if dataset in [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Reddit",
        "Reddit2",
        "Flickr",
        "Yelp",
        "AmazonProducts",
        "PPI",
    ]:
        train_data, val_data, test_data = get_data_SAGE(config)
    else:
        logger.info("TODO: support custom dataset.")
        unlabelled_data = None
        raise NotImplementedError

    split = config["vargs"]["split"]

    match split:
        case "train":
            train_data.infer_mask = train_data.train_mask
            return train_data
        case "val":
            val_data.infer_mask = val_data.val_mask
            return val_data
        case "test":
            test_data.infer_mask = test_data.test_mask
            return test_data
        case "unlabelled":
            unlabelled_data.infer_mask = torch.ones(
                unlabelled_data.num_nodes, dtype=int)
            return unlabelled_data


def get_inference_data_SAINT(config):
    pass


def get_inference_data_graph_batch(config):
    pass


def get_inference_loader_SAGE(infer_data: Data, config: dict):
    model_config = config["model_config"]
    num_neighbors = model_config.get("num_neighbors", -1)
    if type(num_neighbors) == int:
        num_neighbors = [num_neighbors] * model_config["num_layers"]
    elif type(num_neighbors) == list:
        num_neighbors = num_neighbors
        assert len(num_neighbors) == model_config["num_layers"]

    params = config["hyperparameters"]

    general_config = config["general_config"]

    logger.info(
        f"\ninference_data={infer_data}")

    if not general_config["sample_when_predict"]:
        logger.warning(
            "sample_when_predict is set to be False. All neighbors will be used for aggregation when doing prediction in validation and testing.")
        num_neighbors = [-1] * model_config["num_layers"]

    infer_loader = NeighborLoader(
        infer_data,
        num_neighbors=num_neighbors.copy(),
        batch_size=params["batch_size"],
        input_nodes=infer_data.infer_mask,
        num_workers=general_config["num_workers"],
        persistent_workers=general_config["persistent_workers"],
    )

    return infer_data, infer_loader


def get_inference_loader_SAINT(data: Data, config: dict):
    pass


def get_inference_loader_no_sampling(infer_data: Data, config: dict):
    logger.warning(
        "Sampling strategy is set to be None. Full graph will be used without mini-batching! Batch_size is ignored! ")
    
    logger.info(
        f"\ninference_data={infer_data}")

    infer_loader = DataLoader([infer_data])
    
    return infer_data, infer_loader


def get_inference_loader_graph_batch(data: Data, config: dict):
    pass


def get_inference_loader(config):
    sampling_strategy = config["general_config"]["sampling_strategy"]
    if sampling_strategy == 'SAGE':
        return get_inference_loader_SAGE(get_inference_data_SAGE(config), config)
    elif sampling_strategy == 'SAINT':
        return get_inference_loader_SAINT(get_inference_data_SAINT(config), config)
    elif sampling_strategy == 'GraphBatching':
        return get_inference_loader_graph_batch(get_inference_data_graph_batch(config), config)
    elif sampling_strategy == 'None' or sampling_strategy == None:
        return get_inference_loader_no_sampling(get_inference_data_SAGE(config), config)
