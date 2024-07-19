import torch
import torch_geometric.transforms as T

from torch_geometric.data import Data, Batch
from torch_geometric.loader import (NeighborLoader, LinkNeighborLoader,
                                    DataLoader)

from loguru import logger
import pprint


def merge_from_data_list(data_list):
    batch_data = Batch.from_data_list(data_list)
    data = Data(x=batch_data.x,
                edge_index=batch_data.edge_index,
                y=batch_data.y)
    # TODO: also support edge features and edge lable.

    return data


def get_data_SAGE(config):
    dataset_transform = T.Compose([T.NormalizeFeatures()])
    batch_transform = None

    dataset = config["dataset"]
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid('dataset',
                            dataset,
                            split='public',
                            transform=dataset_transform)
    elif dataset == "Reddit":
        from torch_geometric.datasets import Reddit
        dataset = Reddit('dataset/Reddit', transform=dataset_transform)
    elif dataset == "Reddit2":
        from torch_geometric.datasets import Reddit2
        dataset = Reddit2('dataset/Reddit2', transform=dataset_transform)
    elif dataset == "Flickr":
        from torch_geometric.datasets import Flickr
        dataset = Flickr('dataset/Flickr', transform=dataset_transform)
    elif dataset == " Yelp":
        from torch_geometric.datasets import Yelp
        dataset = Yelp('dataset/Yelp', transform=dataset_transform)
    elif dataset == "AmazonProducts":
        from torch_geometric.datasets import AmazonProducts
        dataset = AmazonProducts('dataset/AmazonProducts',
                                 transform=dataset_transform)
    elif dataset == "PPI":
        from torch_geometric.datasets import PPI
        dataset = [
            merge_from_data_list(PPI('dataset/PPI', split='train')),
            merge_from_data_list(PPI('dataset/PPI', split='val')),
            merge_from_data_list(PPI('dataset/PPI', split='test')),
        ]
    elif dataset.startswith("AMLworld"):
        AMLworld_config = config["AMLworld_config"]
        config["dataset_config"].update(AMLworld_config)
        logger.info(
            f"AMLworld configuration: {pprint.pformat(AMLworld_config)}")
        dataset_config = config["dataset_config"]
        task_type = AMLworld_config["task_type"]
        if task_type.endswith("NC"):
            readout = "node"
        elif task_type.endswith("EC"):
            readout = "edge"
        else:
            raise NotImplementedError

        from pyg.custom_dataset.AMLworld import (
            AMLworld, AddEgoIds_for_NeighborLoader,
            AddEgoIds_for_LinkNeighborLoader)
        option = dataset.partition("-")[-1]
        dataset = []
        for split in ["train", "val", "test"]:
            dataset.append(
                AMLworld('dataset/AMLworld',
                         opt=option,
                         split=split,
                         load_time_stamp=AMLworld_config["add_time_stamp"],
                         load_ports=AMLworld_config["add_port"],
                         load_time_delta=AMLworld_config["add_time_delta"],
                         ibm_split=AMLworld_config["ibm_split"],
                         force_reload=AMLworld_config["force_reload"],
                         verbose=False,
                         readout=readout)[0])

        if AMLworld_config["add_egoID"]:
            if task_type.endswith("NC"):
                AddEgoIds = AddEgoIds_for_NeighborLoader
            elif task_type.endswith("EC"):
                AddEgoIds = AddEgoIds_for_LinkNeighborLoader
            else:
                raise ValueError("Unsupported task type for add_egoID!")
            batch_transform = AddEgoIds()
        dataset_config["num_node_features"] = dataset[0].x.shape[1] + \
            int(AMLworld_config["add_egoID"])
        dataset_config["num_edge_features"] = dataset[0].edge_attr.shape[1]
    else:
        raise NotImplementedError('Unsupported dataset.')

    general_config = config["general_config"]

    # Node Classification
    task_type = config["dataset_config"]["task_type"]
    if task_type in ["single-label-NC", "multi-label-NC"]:
        # For dataset containing one graph and indicate split by mask
        if len(dataset) == 1:
            data = dataset[0]
            if general_config["framework"] == "transductive":
                logger.info("Using data split for transductive training.")
                train_data = data
                val_data = data
                test_data = data

            elif general_config["framework"] == "inductive":
                if general_config["SAGE_inductive_option"] in [
                        "default", "strict"
                ]:
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

        # For dataset with splits as different graphs
        else:
            train_data, val_data, test_data = dataset

            train_data.train_mask = torch.ones(train_data.num_nodes,
                                               dtype=bool)
            val_data.val_mask = torch.ones(val_data.num_nodes, dtype=bool)
            test_data.test_mask = torch.ones(test_data.num_nodes, dtype=bool)

    elif task_type in ["single-label-EC"]:
        # For dataset containing one graph and indicate split by mask
        if len(dataset) == 1:
            data = dataset[0]
            if general_config["framework"] == "transductive":
                logger.info("Using data split for transductive training.")
                train_data = data
                val_data = data
                test_data = data

            elif general_config["framework"] == "inductive":
                if general_config["SAGE_inductive_option"] in [
                        "default", "strict"
                ]:
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

        # For dataset with splits as different graphs
        else:
            train_data, val_data, test_data = dataset

            train_data.train_mask = torch.ones(train_data.num_edges,
                                               dtype=bool)
            val_data.val_mask = torch.ones(val_data.num_edges, dtype=bool)
            test_data.test_mask = torch.ones(test_data.num_edges, dtype=bool)

    else:
        raise NotImplementedError("Unsupported task type!")

    if config["model_config"].get("reverse_mp", False):
        if train_data.is_undirected():
            raise ValueError(
                "Reverse message passing should not be "
                "applied on undirected graph."
            )
        for data in [train_data, val_data, test_data]:
            if not hasattr(data, "rev_edge_attr"):
                if hasattr(data, "edge_attr"):
                    if hasattr(data, "edge_features_colID"):
                        rev_edge_features_colID = {}
                        edge_features_colID = data.edge_features_colID
                        for feature, id in edge_features_colID.items():
                            if feature.startswith("In-"):
                                rev_edge_features_colID[
                                    feature] = edge_features_colID[
                                        f"Out-{feature[3:]}"]
                            elif feature.startswith("Out-"):
                                rev_edge_features_colID[
                                    feature] = edge_features_colID[
                                        f"In-{feature[4:]}"]
                            else:
                                rev_edge_features_colID[feature] = id

                        data.rev_edge_features_colID = rev_edge_features_colID

                        rev_ID = []
                        for feature, id in sorted(
                                rev_edge_features_colID.items(),
                                key=lambda x: x[1]):
                            rev_ID.append(edge_features_colID[feature])
                        data.rev_edge_attr = data.edge_attr[:, rev_ID]

                    else:
                        data.rev_edge_attr = data.edge_attr
        logger.info("Reverse edges added.")

    return train_data, val_data, test_data, batch_transform


def get_data_SAINT(config):
    raise NotImplementedError('Not Implemented.')


def get_data_graph_batch(config):
    batch_transform = None
    if config["dataset"] == "PPI":
        from torch_geometric.datasets import PPI
        train_dataset = PPI('dataset/PPI', split='train')
        val_dataset = PPI('dataset/PPI', split='val')
        test_dataset = PPI('dataset/PPI', split='test')

    return train_dataset, val_dataset, test_dataset, batch_transform


def get_loader_SAGE(train_data, val_data, test_data, transform, config):
    model_config = config["model_config"]
    params = config["hyperparameters"]

    num_neighbors = model_config.get("num_neighbors", -1)
    if isinstance(num_neighbors, int):
        num_neighbors = [num_neighbors] * model_config["num_layers"]
    elif isinstance(num_neighbors, list):
        num_neighbors = num_neighbors
        assert len(num_neighbors) == model_config["num_layers"]

    general_config = config["general_config"]

    logger.info(
        f"\ntrain_data={train_data}\n"
        f"val_data={val_data}\ntest_data={test_data}"
    )

    task_type = config["dataset_config"]["task_type"]
    if task_type in ["single-label-NC", "multi-label-NC"]:
        train_loader = NeighborLoader(
            train_data,
            num_neighbors=num_neighbors.copy(),
            batch_size=params["batch_size"],
            input_nodes=train_data.train_mask,
            num_workers=general_config["num_workers"],
            persistent_workers=general_config["persistent_workers"],
            transform=transform,
        )

        if not general_config["sample_when_predict"]:
            logger.warning(
                "sample_when_predict is set to be False. All neighbors will "
                "be used for aggregation when doing prediction in validation "
                "and testing."
            )
            num_neighbors = [-1] * model_config["num_layers"]

        val_loader = NeighborLoader(
            val_data,
            num_neighbors=num_neighbors,
            batch_size=params["batch_size"],
            input_nodes=val_data.val_mask,
            num_workers=general_config["num_workers"],
            persistent_workers=general_config["persistent_workers"],
            transform=transform,
        )

        test_loader = NeighborLoader(
            test_data,
            num_neighbors=num_neighbors,
            batch_size=params["batch_size"],
            input_nodes=test_data.test_mask,
            num_workers=general_config["num_workers"],
            persistent_workers=general_config["persistent_workers"],
            transform=transform,
        )

    elif task_type in ["single-label-EC"]:
        train_loader = LinkNeighborLoader(
            train_data,
            num_neighbors=num_neighbors,
            batch_size=params["batch_size"],
            edge_label_index=train_data.edge_index[:, train_data.train_mask],
            edge_label=train_data.y,
            transform=transform,
        )

        val_loader = LinkNeighborLoader(
            val_data,
            num_neighbors=num_neighbors,
            batch_size=params["batch_size"],
            edge_label_index=val_data.edge_index[:, val_data.val_mask],
            edge_label=val_data.y,
            transform=transform,
        )

        test_loader = LinkNeighborLoader(
            test_data,
            num_neighbors=num_neighbors,
            batch_size=params["batch_size"],
            edge_label_index=test_data.edge_index[:, test_data.test_mask],
            edge_label=test_data.y,
            transform=transform,
        )

    return train_loader, val_loader, test_loader


def get_loader_SAINT(data: Data, config):
    raise NotImplementedError("Not implemented.")


def get_loader_no_sampling(train_data, val_data, test_data, transform, config):
    logger.warning(
        "Sampling strategy is set to be None. Full graph will be "
        "used without mini-batching! Batch_size is ignored!"
    )

    logger.info(
        f"\ntrain_data={train_data}\n"
        f"val_data={val_data}\ntest_data={test_data}"
    )

    if transform:
        train_data = transform(train_data)
        val_data = transform(val_data)
        test_data = transform(test_data)

    train_loader = DataLoader([train_data])
    val_loader = DataLoader([val_data])
    test_loader = DataLoader([test_data])

    train_loader.data = train_data
    val_loader.data = val_data
    test_loader.data = test_data

    return train_loader, val_loader, test_loader


def get_loader_graph_batch(train_dataset, val_dataset, test_dataset, transform,
                           config):
    batch_size = config["hyperparameters"]["batch_size"]
    general_config = config["general_config"]

    if general_config["sampling_strategy"] == "GraphBatching" and config[
            "dataset_config"]["task_type"].endswith("-EC"):
        raise NotImplementedError(
            "Graph Batching is not implemented for edge classification task!")

    if transform:
        train_dataset = transform(train_dataset)
        val_dataset = transform(val_dataset)
        test_dataset = transform(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=general_config["num_workers"],
        persistent_workers=general_config["persistent_workers"],
        shuffle=True,
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

    # TODO: save data into loader

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
    elif sampling_strategy == 'None' or sampling_strategy is None:
        return get_loader_no_sampling(*get_data_SAGE(config), config)


def get_inference_data_SAGE(config):
    dataset = config["dataset"]
    batch_transform = None

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
        train_data, val_data, test_data, batch_transform = get_data_SAGE(
            config)
    else:
        logger.info("TODO: support custom dataset.")
        unlabelled_data = None
        raise NotImplementedError

    split = config["vargs"]["split"]

    match split:
        case "train":
            train_data.infer_mask = train_data.train_mask
            return train_data, batch_transform
        case "val":
            val_data.infer_mask = val_data.val_mask
            return val_data, batch_transform
        case "test":
            test_data.infer_mask = test_data.test_mask
            return test_data, batch_transform
        case "unlabelled":
            unlabelled_data.infer_mask = torch.ones(unlabelled_data.num_nodes,
                                                    dtype=int)
            return unlabelled_data, batch_transform


def get_inference_data_SAINT(config):
    pass


def get_inference_data_graph_batch(config):
    pass


def get_inference_loader_SAGE(infer_data: Data, transform, config: dict):
    model_config = config["model_config"]
    num_neighbors = model_config.get("num_neighbors", -1)
    if isinstance(num_neighbors, int):
        num_neighbors = [num_neighbors] * model_config["num_layers"]
    elif isinstance(num_neighbors, list):
        num_neighbors = num_neighbors
        assert len(num_neighbors) == model_config["num_layers"]

    params = config["hyperparameters"]

    general_config = config["general_config"]

    logger.info(f"\ninference_data={infer_data}")

    if not general_config["sample_when_predict"]:
        logger.warning(
            "sample_when_predict is set to be False. "
            "All neighbors will be used for aggregation "
            "when doing prediction in validation and testing."
        )
        num_neighbors = [-1] * model_config["num_layers"]

    infer_loader = NeighborLoader(
        infer_data,
        num_neighbors=num_neighbors.copy(),
        batch_size=params["batch_size"],
        input_nodes=infer_data.infer_mask,
        num_workers=general_config["num_workers"],
        persistent_workers=general_config["persistent_workers"],
        transform=transform,
    )

    return infer_loader


def get_inference_loader_SAINT(data: Data, config: dict):
    pass


def get_inference_loader_no_sampling(infer_data: Data, transform,
                                     config: dict):
    logger.warning(
        "Sampling strategy is set to be None. "
        "Full graph will be used without mini-batching! "
        "Batch_size is ignored!"
    )

    logger.info(f"\ninference_data={infer_data}")

    if transform:
        infer_data = transform(infer_data)

    infer_loader = DataLoader([infer_data])
    infer_loader.data = infer_data

    return infer_loader


def get_inference_loader_graph_batch(data: Data, config: dict):
    pass


def get_inference_loader(config):
    sampling_strategy = config["general_config"]["sampling_strategy"]
    if sampling_strategy == 'SAGE':
        return get_inference_loader_SAGE(*get_inference_data_SAGE(config),
                                         config)
    elif sampling_strategy == 'SAINT':
        return get_inference_loader_SAINT(*get_inference_data_SAINT(config),
                                          config)
    elif sampling_strategy == 'GraphBatching':
        return get_inference_loader_graph_batch(
            *get_inference_data_graph_batch(config), config)
    elif sampling_strategy == 'None' or sampling_strategy is None:
        return get_inference_loader_no_sampling(
            *get_inference_data_SAGE(config), config)
