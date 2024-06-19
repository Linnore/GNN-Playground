import torch
import torch_geometric.transforms as T

from torch_geometric.data import Data, Batch
from torch_geometric.loader import NeighborLoader, DataLoader

from loguru import logger


def merge_from_data_list(data_list):
    batch_data = Batch.from_data_list(data_list)
    data =  Data(
        x=batch_data.x,
        edge_index=batch_data.edge_index,
        y=batch_data.y
    )
    # TODO: also support edge features and edge lable.
    
    return data


def get_data_SAGE(config):
    transform = T.Compose([T.NormalizeFeatures()])
    if config["dataset"] in ['Cora', 'CiteSeer', 'PubMed']:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid('dataset', config["dataset"], split='public', transform=transform)
    elif config["dataset"] == "Reddit":
        from torch_geometric.datasets import Reddit
        dataset = Reddit('dataset/Reddit', transform=transform)
    elif config["dataset"] == "Reddit2":
        from torch_geometric.datasets import Reddit2
        dataset = Reddit2('dataset/Reddit2', transform=transform)
    elif config["dataset"] == "Flickr":
        from torch_geometric.datasets import Flickr
        dataset = Flickr('dataset/Flickr', transform=transform)
    elif config["dataset"] == " Yelp":
        from torch_geometric.datasets import Yelp
        dataset = Yelp('dataset/Yelp', transform=transform)
    elif config["dataset"] == "AmazonProducts":
        from torch_geometric.datasets import AmazonProducts
        dataset = AmazonProducts('dataset/AmazonProducts', transform=transform)
    elif config["dataset"] == "PPI":
        from torch_geometric.datasets import PPI
        dataset = [
            merge_from_data_list(PPI('dataset/PPI', split='train')),
            merge_from_data_list(PPI('dataset/PPI', split='val')),
            merge_from_data_list(PPI('dataset/PPI', split='test')),
        ]
    else:
        logger.exception('Unsupported dataset.')
        
    general_config = config["general_config"]
    
    # For dataset containing one graph
    if len(dataset) == 1:
        data = dataset[0]
        if general_config["framework"] == "transductive":
            logger.info("Using data split for transductive training.")
            train_data = data
            val_data = data
            test_data = data
            general_config.pop("SAGE_inductive_option")

        elif general_config["framework"] == "inductive":
            if general_config["SAGE_inductive_option"] in ["default", "strict"]:
                logger.info("Using data split for strict inductive learning.")
                train_data = data.subgraph(data.train_mask)
                val_data = data.subgraph(data.val_mask)
                test_data = data.subgraph(data.test_mask)
            elif general_config["SAGE_inductive_option"] == "soft":
                logger.info("Using data split for non-strict inductive learning.")
                train_data = data.subgraph(data.train_mask)
                val_data = data
                test_data = data
                
    # For dataset containing multiple graphs. Strictly inductive learning by default.
    else:
        train_data, val_data, test_data = dataset
        
        train_data.train_mask = torch.ones(train_data.num_nodes, dtype=bool)
        val_data.val_mask = torch.ones(val_data.num_nodes, dtype=bool)
        test_data.test_mask = torch.ones(test_data.num_nodes, dtype=bool)
        
    
    return train_data, val_data, test_data
                


def get_data_SAINT(config):
    dataset_config = config["dataset_collections"][config["dataset"]]
    logger.exception('Not Implemented.')


def get_data_graph_batch(config):
    if config["dataset"] == "PPI":
        from torch_geometric.datasets import PPI
        train_dataset = PPI('dataset/PPI', split='train')
        val_dataset = PPI('dataset/PPI', split='val')
        test_dataset = PPI('dataset/PPI', split='test')
        
    return train_dataset, val_dataset, test_dataset
        
    

def get_loader_SAGE(train_data, val_data, test_data, config):
    model_config = config["model_collections"][config["model"]]
    num_neighbors = model_config.pop("num_neighbors")
    if type(num_neighbors) == int:
        num_neighbors = [num_neighbors] * model_config["num_layers"]
    elif type(num_neighbors) == list:
        num_neighbors = num_neighbors
        assert len(num_neighbors) == model_config["num_layers"]

    params = config["hyperparameters"]
    
    general_config = config["general_config"]

    logger.info(
        f"\ntrain_data={train_data}\nval_data={val_data}\ntest_data={test_data}")

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

    return train_loader, val_loader, test_loader


def get_loader_SAINT(data: Data, config):
    logger.exception("Not implemented.")


def get_loader_no_sampling(train_data, val_data, test_data, config):
    model_config = config["model_collections"][config["model"]]

    logger.warning(
        "Sampling strategy is set to be None. Full graph will be used without mini-batching! Batch_size is ignored! ")
    num_neighbors = [-1] * model_config["num_layers"]
    model_config.pop("num_neighbors", None)

    general_config = config["general_config"]

    logger.info(
        f"\ntrain_data={train_data}\nval_data={val_data}\ntest_data={test_data}")

    train_loader = NeighborLoader(
        train_data,
        num_neighbors=num_neighbors,
        batch_size=train_data.num_nodes,
        input_nodes=train_data.train_mask,
        num_workers=general_config["num_workers"],
        persistent_workers=general_config["persistent_workers"],
    )

    val_loader = NeighborLoader(
        val_data,
        num_neighbors=num_neighbors,
        batch_size=val_data.num_nodes,
        input_nodes=val_data.val_mask,
        num_workers=general_config["num_workers"],
        persistent_workers=general_config["persistent_workers"],
    )

    test_loader = NeighborLoader(
        test_data,
        num_neighbors=num_neighbors,
        batch_size=test_data.num_nodes,
        input_nodes=test_data.test_mask,
        num_workers=general_config["num_workers"],
        persistent_workers=general_config["persistent_workers"],
    )

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
