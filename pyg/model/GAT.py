import torch
import torch.nn.functional as F

from loguru import logger

from torch.nn import Linear, Identity, ModuleList

from torch_geometric.nn.models import GAT as GAT_Base
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn import GATConv, GATv2Conv


class GAT_PyG(GAT_Base):
    def __init__(self, config={}, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.config = config


class GAT_Custom(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_node_channels_per_head: int | list[int],
                 num_layers: int, out_channels: int,
                 heads: int | list[int] = 8,
                 output_heads: int = 1,
                 v2: bool = False,
                 dropout: float = 0.6,
                 jk=None,
                 skip_connection: bool = False,
                 config={},
                 *args,
                 **kwargs):
        super().__init__()
        self.config = config
        
        if v2:
            Conv = GATv2Conv
        else:
            Conv = GATConv

        if type(hidden_node_channels_per_head) == int:
            hidden_node_channels_per_head = [
                hidden_node_channels_per_head] * (num_layers-1)

        if type(heads) == int:
            heads = [heads] * (num_layers-1)

        self.conv = ModuleList()
        self.conv.append(
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

        self.jk_mode = jk
        if not self.jk_mode in ["cat", None]:
            logger.exception(NotImplementedError(
                "JK mode not implemented. Only support concat JK for now!"))
        if self.jk_mode != None:
            self.jk = JumpingKnowledge(jk)
            jk_in_channels = out_channels
            for i in range(len(heads)):
                jk_in_channels += heads[i] * hidden_node_channels_per_head[i]
            self.jk_linear = Linear(jk_in_channels, out_channels)
            
        self.skip_connection = skip_connection
        if self.skip_connection:
            self.skip_proj = ModuleList()
            self.skip_proj.append(
                self.get_skip_proj(in_channels, hidden_node_channels_per_head[0]*heads[0])
            )
            for i in range(1, num_layers-1):
                self.skip_proj.append(
                    self.get_skip_proj(
                        hidden_node_channels_per_head[i-1]*heads[i-1],
                        hidden_node_channels_per_head[i]*heads[i]
                    )
                )
            self.skip_proj.append(
                self.get_skip_proj(hidden_node_channels_per_head[-1]*heads[-1], out_channels)
            )
            

    def get_skip_proj(self, in_channels, out_channels):
        if in_channels == out_channels:
            return Identity()
        else:
            return Linear(in_channels, out_channels)

    def reset_parameters(self):
        for conv in self.conv:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        xs = []
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip_connection:
                residual = self.skip_proj[i](x)
            x = self.conv[i](x, edge_index)
            x = x + residual if self.skip_connection else x
            x = F.elu(x)
            if self.jk_mode != None:
                xs.append(x)
                
        if self.jk_mode != None:
            x = self.jk(xs)
            x = self.jk_linear(x)
        return x
