import torch
import torch.nn.functional as F

from loguru import logger

from torch.nn import Linear, Identity, ModuleList

from torch_geometric.nn.models import GIN as GIN_Base
from torch_geometric.nn.models import JumpingKnowledge, MLP
from torch_geometric.nn import GINConv, GINEConv


class GIN_PyG(GIN_Base):
    def __init__(self, config={}, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.config = config


class GIN_Custom(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_node_channels: int | list[int],
                 num_layers: int, out_channels: int,
                 num_MLP_layers: int = 2,
                 GINE: bool = False,
                 dropout: float = 0.6,
                 jk=None,
                 skip_connection: bool = False,
                 config={},
                 *args,
                 **kwargs):
        super().__init__()
        self.config = config
        
        if GINE:
            Conv = GINEConv
        else:
            Conv = GINConv
        
        self.dropout = dropout
        self.num_layers = num_layers
        
        if type(hidden_node_channels) == int:
            hidden_node_channels = [hidden_node_channels] * (num_layers-1)

        self.conv = ModuleList()
        self.conv.append(
            Conv(nn=self.init_MLP(
                in_channels, 
                hidden_node_channels[0], 
                num_MLP_layers
            ))
        )
        for i in range(1, num_layers-1):
            self.conv.append(
                Conv(self.init_MLP(
                    hidden_node_channels[i-1], 
                    hidden_node_channels[i], 
                    num_MLP_layers
                ))
            )
        self.conv.append(Conv(self.init_MLP(
                    hidden_node_channels[-1], 
                    out_channels, 
                    num_MLP_layers
        )))


        self.jk_mode = jk
        if not self.jk_mode in ["cat", None]:
            logger.exception(NotImplementedError(
                "JK mode not implemented. Only support concat JK for now!"))
        if self.jk_mode != None:
            self.jk = JumpingKnowledge(jk)
            jk_in_channels = out_channels
            for i in range(len(hidden_node_channels)):
                jk_in_channels += hidden_node_channels[i]
            self.jk_linear = Linear(jk_in_channels, out_channels)
            
        self.skip_connection = skip_connection
        if self.skip_connection:
            self.skip_proj = ModuleList()
            self.skip_proj.append(
                self.get_skip_proj(in_channels, hidden_node_channels[0])
            )
            for i in range(1, num_layers-1):
                self.skip_proj.append(
                    self.get_skip_proj(
                        hidden_node_channels[i-1],
                        hidden_node_channels[i]
                    )
                )
            self.skip_proj.append(
                self.get_skip_proj(hidden_node_channels[-1], out_channels)
            )
            
    def init_MLP(self, in_channels: int, out_channels: int, num_MLP_layers):
        channel_list = [in_channels]
        for i in range(num_MLP_layers):
            channel_list.append(out_channels)
        mlp = MLP(channel_list, dropout=self.dropout, norm=None)
        return mlp
    
    def get_skip_proj(self, in_channels, out_channels):
        if in_channels == out_channels:
            return Identity()
        else:
            return Linear(in_channels, out_channels)

    def reset_parameters(self):
        for conv in self.conv:
            conv.reset_parameters()
        if self.jk_mode:
            for nn in self.jk_linear:
                nn.reset_parameters()
        if self.skip_connection:
            for nn in self.skip_proj:
                nn.register_parameter()

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
