from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from loguru import logger

def get_data_from_Planetoid(config):
    from torch_geometric.datasets import Planetoid
    dataset = config["dataset"]
    data = Planetoid('dataset', dataset, split='public')[0]
    if dataset == 'Cora':
        config["num_node_features"] = 1433
        config["num_classes"] = 7
    elif dataset == 'CiteSeer':
        config["num_node_features"] = 3703
        config["num_classes"] = 6
    elif dataset == 'PubMed':
        config["num_node_features"] = 500
        config["num_classes"] = 3
    return data
    


def get_data_SAGE(config):
    if config["dataset"] in ['Cora', 'CiteSeer', 'PubMed']:
        data = get_data_from_Planetoid(config)
    else:
        logger.exception('Unsupported dataset.')

    return data


def get_data_SAINT(config):
    logger.exception('Not Implemented.')


def get_loader_SAGE(data:Data, config):
    
    train_loader = NeighborLoader(
        data, 
        num_neighbors=[config["num_neighbors"]] * config["num_layers"],
        batch_size = config["batch_size"],
        input_nodes=data.train_mask
    )
    
    val_loader = NeighborLoader(
        data, 
        num_neighbors=[config["num_neighbors"]] * config["num_layers"],
        batch_size = config["batch_size"],
        input_nodes=data.val_mask
    )
    
    test_loader = NeighborLoader(
        data, 
        num_neighbors=[config["num_neighbors"]] * config["num_layers"],
        batch_size = config["batch_size"],
        input_nodes=data.test_mask
    )
    
    return train_loader, val_loader, test_loader


def get_loader_SAINT(data:Data, config):
    logger.exception("Not implemented.")
    
    
def get_data(config):
    if config["inductive_type"] == 'SAGE':
        return get_data_SAGE(config)
    elif config["inductive_type"] == 'SAINT':
        return get_data_SAINT(config)
    
    
def get_loader(config):
    if config["inductive_type"] == 'SAGE':
        return get_loader_SAGE(get_data_SAGE(config), config)
    elif config["inductive_type"] == 'SAINT':
        return get_loader_SAINT(get_data_SAINT(config), config)
    
    
    
    
    
    
    

    
    