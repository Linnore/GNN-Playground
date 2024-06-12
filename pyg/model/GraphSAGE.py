import torch_geometric.nn


from loguru import logger
from torch import softmax
from torch_geometric.nn.models import GraphSAGE as GraphSAGE_Default



class GraphSAGEe(GraphSAGE_Default):
    def __init__(self, *args, **kwargs):
        super.__init__(self, *args, **kwargs)
        logger.warnning("GraphSAGE with edge feature update is not implemented.")   
