import torch

from loguru import logger
from torch_geometric.nn.models import GAT as GAT_PyG
from torch_geometric.nn.models.basic_gnn import BasicGNN



class GATe(GAT_PyG):
    def __init__(self, *args, **kwargs):
        super.__init__(self, *args, **kwargs)
        raise NotImplementedError
        
class GAT_Custom(BasicGNN):
    def __init__(self):
        raise NotImplementedError