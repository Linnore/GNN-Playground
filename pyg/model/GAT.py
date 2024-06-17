import torch
import torch.nn.functional as F

from loguru import logger
from torch_geometric.nn.models import GAT as GAT_PyG
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn import GATConv, GATv2Conv


class GATe(GAT_PyG):
    def __init__(self, *args, **kwargs):
        super.__init__(self, *args, **kwargs)
        raise NotImplementedError


class GAT_Custom(torch.nn.Module):
    def __init__(self, in_channels, hidden_node_channels_per_head: int | list[int], num_layers: int, out_channels: int, heads: int | list[int] = 8, output_heads: int = 1, v2: bool = False, dropout: float = 0.6, *args, **kwargs):
        super().__init__()
        if v2:
            Conv = GATv2Conv
        else:
            Conv = GATConv

        if type(hidden_node_channels_per_head) == int:
            hidden_node_channels_per_head = [
                hidden_node_channels_per_head] * (num_layers-1)

        if type(heads) == int:
            heads = [heads] * (num_layers-1)

        self.conv = torch.nn.Sequential(
            Conv(in_channels,
                 hidden_node_channels_per_head[0], heads[0], dropout=dropout)
        )
        for i in range(1, num_layers-1):
            self.conv.append(
                Conv(hidden_node_channels_per_head[i-1]*heads[i-1],
                     hidden_node_channels_per_head[i], heads[i], dropout=dropout)
            )
        self.conv.append(Conv(
            hidden_node_channels_per_head[-1] * heads[-1], out_channels, output_heads, concat=False, dropout=dropout))

        self.dropout = dropout
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.conv:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers-1):
            x = F.elu(self.conv[i](x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv[-1](x, edge_index)
        return x
