import torch
import torch.nn.functional as F

from loguru import logger

from torch.nn import Linear, Identity, ModuleList

from torch_geometric.nn.models import PNA as PNA_Base
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn import PNAConv


class PNA_PyG(PNA_Base):
    def __init__(self, 
                 deg,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 config={}, 
                 *args, 
                 **kwargs):
        kwargs["deg"] = deg
        kwargs["aggregators"] = aggregators
        kwargs["scalers"] = scalers
        super().__init__(*args, **kwargs)
        self.config = config


class PNA_Custom(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels: int | list[int],
                 num_layers: int, out_channels: int,
                 deg,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 dropout: float = 0.6,
                 jk=None,
                 skip_connection: bool = False,
                 config={},
                 *args,
                 **kwargs):
        super().__init__()
        self.config = config

        Conv = PNAConv

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        if type(hidden_channels) == int:
            hidden_channels = [hidden_channels] * (num_layers-1)

        self.conv = ModuleList()
        self.conv.append(
            Conv(in_channels,
                 hidden_channels[0],
                 aggregators,
                 scalers,
                 deg=deg
                 ))
        for i in range(1, num_layers-1):
            self.conv.append(
                Conv(hidden_channels[i-1],
                     hidden_channels[i],
                     aggregators,
                     scalers,
                     deg=deg
                     ))
        self.conv.append(
            Conv(hidden_channels[-1],
                 out_channels,
                 aggregators,
                 scalers,
                 deg=deg
                 ))

        self.jk_mode = jk
        if not self.jk_mode in ["cat", None]:
            raise NotImplementedError(NotImplementedError(
                "JK mode not implemented. Only support concat JK for now!"))
        if self.jk_mode != None:
            self.jk = JumpingKnowledge(jk)
            jk_in_channels = out_channels
            for i in range(len(hidden_channels)):
                jk_in_channels += hidden_channels[i]
            self.jk_linear = Linear(jk_in_channels, out_channels)

        self.skip_connection = skip_connection
        if self.skip_connection:
            self.skip_proj = ModuleList()
            self.skip_proj.append(
                self.get_skip_proj(in_channels, hidden_channels[0])
            )
            for i in range(1, num_layers-1):
                self.skip_proj.append(
                    self.get_skip_proj(
                        hidden_channels[i-1],
                        hidden_channels[i]
                    )
                )
            self.skip_proj.append(
                self.get_skip_proj(hidden_channels[-1], out_channels)
            )

    def get_skip_proj(self, in_channels, out_channels):
        if in_channels == out_channels:
            return Identity()
        else:
            return Linear(in_channels, out_channels)

    def reset_parameters(self):
        for conv in self.conv:
            conv.reset_parameters()
        if self.jk_mode:
            self.jk_linear.reset_parameters()
        if self.skip_connection:
            for nn in self.skip_proj:
                if isinstance(nn, Linear):
                    nn.reset_parameters()

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
