from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from loguru import logger

def get_data_from_Planetoid(name, dataset_config):
    from torch_geometric.datasets import Planetoid
    data = Planetoid('dataset', name, split='public')[0]
    if name == 'Cora':
        dataset_config["num_node_features"] = 1433
        dataset_config["num_classes"] = 7
    elif name == 'CiteSeer':
        dataset_config["num_node_features"] = 3703
        dataset_config["num_classes"] = 6
    elif name == 'PubMed':
        dataset_config["num_node_features"] = 500
        dataset_config["num_classes"] = 3
    return data
    


def get_data_SAGE(config):
    dataset_config = config["dataset_collections"][config["dataset"]]
    if config["dataset"] in ['Cora', 'CiteSeer', 'PubMed']:
        data = get_data_from_Planetoid(config["dataset"], dataset_config)
    else:
        logger.exception('Unsupported dataset.')

    return data


def get_data_SAINT(config):
    dataset_config = config["dataset_collections"][config["dataset"]]
    logger.exception('Not Implemented.')


def get_loader_SAGE(data:Data, config):
    model_config = config["model_collections"][config["model"]]
    if type(model_config["num_neighbors"]) == int:
        num_neighbors = [model_config["num_neighbors"]] * model_config["num_layers"]
    elif type(model_config["num_neighbors"]) == list:
        num_neighbors = model_config["num_neighbors"]
        assert len(num_neighbors) == model_config["num_layers"]
    
    train_loader = NeighborLoader(
        data, 
        num_neighbors=num_neighbors,
        batch_size = config["hyperparameters"]["batch_size"],
        input_nodes=data.train_mask
    )
    
    val_loader = NeighborLoader(
        data, 
        num_neighbors=num_neighbors,
        batch_size = config["hyperparameters"]["batch_size"],
        input_nodes=data.val_mask
    )
    
    test_loader = NeighborLoader(
        data, 
        num_neighbors=num_neighbors,
        batch_size = config["hyperparameters"]["batch_size"],
        input_nodes=data.test_mask
    )
    
    logger.debug(f"{len(train_loader)}")
    
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
    
    
    
    
    
    
    

    
    