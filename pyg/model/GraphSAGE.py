from loguru import logger
from torch_geometric.nn.models import GraphSAGE as GraphSAGE_Base



class GraphSAGE_PyG(GraphSAGE_Base):
    def __init__(self, config={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

