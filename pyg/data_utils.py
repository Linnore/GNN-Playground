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
    elif config["dataset"] == "Reddit":
        from torch_geometric.datasets import Reddit
        data = Reddit('dataset/Reddit')[0]
    elif config["dataset"] == "Reddit2":
        from torch_geometric.datasets import Reddit2
        data = Reddit2('dataset/Reddit2')[0]
    elif config["dataset"] == "Flickr":
        from torch_geometric.datasets import Flickr
        data = Flickr('dataset/Flickr')[0]
    elif config["dataset"] == " Yelp":
        from torch_geometric.datasets import Yelp
        data = Yelp('dataset/Yelp')[0]
    elif config["dataset"] == "AmazonProducts":
        from torch_geometric.datasets import AmazonProducts
        data = AmazonProducts('dataset/AmazonProducts')
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
    
    general_config = config["general_config"]
    params = config["hyperparameters"]
    
    if general_config["framework"] == "transductive":
        logger.info("Using data split for transductive training.")
        train_data = data
        val_data = data
        test_data = data
        general_config.pop("SAGE_option")
        
    elif general_config["framework"] == "inductive":
        if general_config["SAGE_option"] in ["default", "strict"]:
            logger.info("Using data split for strict inductive learning.")
            train_data = data.subgraph(data.train_mask)
            val_data = data.subgraph(data.val_mask)
            test_data = data.subgraph(data.test_mask)
        elif general_config["SAGE_option"] == "soft":
            logger.info("Using data split for non-strict inductive learning.")
            train_data = data.subgraph(data.train_mask)
            val_data = data
            test_data = data
    
    logger.info(f"\ntrain_data={train_data}\nval_data={val_data}\ntest_data={test_data}")
    
    if general_config["sampling_strategy"] == "None":
        logger.warning("Sampling strategy is set to be None. All neighbors will be used!")
        num_neighbors = [-1] * model_config["num_layers"]
        
    train_loader = NeighborLoader(
        train_data, 
        num_neighbors=num_neighbors,
        batch_size = params["batch_size"],
        input_nodes=train_data.train_mask,
        num_workers=general_config["num_workers"]
    )
    
    if not general_config["sample_when_predict"]:
        logger.warning("sample_when_predict is set to be False. All neighbors will be used for aggregation when doing prediction in validation and testing.")
        logger.warning("Sampling strategy is set to be None. All neighbors will be used!")
        num_neighbors = [-1] * model_config["num_layers"]
    
    val_loader = NeighborLoader(
        val_data, 
        num_neighbors=num_neighbors,
        batch_size = params["batch_size"],
        input_nodes=val_data.val_mask,
        num_workers=general_config["num_workers"]
    )
    
    test_loader = NeighborLoader(
        test_data, 
        num_neighbors=num_neighbors,
        batch_size = params["batch_size"],
        input_nodes=test_data.test_mask,
        num_workers=general_config["num_workers"]
    )
    
    return train_loader, val_loader, test_loader


def get_loader_SAINT(data:Data, config):
    logger.exception("Not implemented.")
    
    
def get_data(config):
    if config["general_config"]["sampling_strategy"] == 'SAGE':
        return get_data_SAGE(config)
    elif config["general_config"]["sampling_strategy"] == 'SAINT':
        return get_data_SAINT(config)
    
    
def get_loader(config):
    if config["general_config"]["sampling_strategy"] == 'SAGE':
        return get_loader_SAGE(get_data_SAGE(config), config)
    elif config["general_config"]["sampling_strategy"] == 'SAINT':
        return get_loader_SAINT(get_data_SAINT(config), config)
    
    
    
    
    
    
    

    
    