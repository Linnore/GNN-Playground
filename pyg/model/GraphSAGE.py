from loguru import logger
from torch_geometric.nn.models import GraphSAGE as GraphSAGE_PyG



class GraphSAGEe(GraphSAGE_PyG):
    def __init__(self, *args, **kwargs):
        super.__init__(self, *args, **kwargs)
        raise NotImplementedError
