import torch
import torch.nn.functional as F

from loguru import logger

from torch.nn import Linear, Identity, ModuleList, Sequential, ReLU, Dropout

from torch_geometric.nn.models import PNA as PNA_Base
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn import PNAConv, BatchNorm


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
    def __init__(
            self,
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

        self.Conv = PNAConv

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.aggregators = aggregators
        self.scalars = scalers
        self.deg = deg

        if type(hidden_channels) == int:
            hidden_channels = [hidden_channels] * (num_layers-1)

        self.convs = ModuleList()
        self.convs.append(
            self.Conv(in_channels,
                 hidden_channels[0],
                 aggregators,
                 scalers,
                 deg=deg
                 ))
        for i in range(1, num_layers-1):
            self.convs.append(
                self.Conv(hidden_channels[i-1],
                     hidden_channels[i],
                     aggregators,
                     scalers,
                     deg=deg
                     ))
        self.convs.append(
            self.Conv(hidden_channels[-1],
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
        for conv in self.convs:
            conv.reset_parameters()
        if self.jk_mode:
            self.jk_linear.reset_parameters()
        if self.skip_connection:
            for nn in self.skip_proj:
                if isinstance(nn, Linear):
                    nn.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        rev_edge_index = kwargs.pop("rev_edge_index", False)
        assert len(kwargs) == 0, "Unexpected arguments!"
        if rev_edge_index:
            raise NotImplementedError
        else:
            self.forward_default(x, edge_index)

    def forward_default(self, x, edge_index):
        xs = []
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip_connection:
                residual = self.skip_proj[i](x)
            conv_out = self.convs[i](x, edge_index)
            x = conv_out + residual if self.skip_connection else conv_out
            x = F.elu(x)
            if self.jk_mode != None:
                xs.append(x)

        if self.jk_mode != None:
            x = self.jk(xs)
            x = self.jk_linear(x)
        return x


class PNAe(PNA_Custom):
    # Adjusted model architecture from https://github.com/IBM/Multi-GNN/blob/252b0252afca109d1d216c411c59ff70753b25fc/models.py#L7
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 edge_update: bool = False,
                 edge_dim=None,
                 batch_norm=True,
                 *args,
                 **kwargs,
                 ):
        super().__init__(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            *args,
            **kwargs,
        )
        
        self.batch_norm = batch_norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_update = edge_update
        
        self.node_emb = Linear(self.in_channels, self.hidden_channels)
        self.edge_emb = Linear(edge_dim, self.hidden_channels)
        
        self.convs = ModuleList()
        self.emlps = ModuleList()
        self.batch_norms = ModuleList()
        
        for _ in range(self.num_layers):
            conv = self.Conv(
                in_channels=hidden_channels, 
                out_channels=hidden_channels,
                aggregators=self.aggregators,
                scalers=self.scalars,
                deg=self.deg,
                edge_dim=hidden_channels,
                towers=5,
                pre_layers=1,
                post_layers=1,
                divide_input=False
            )
            if self.edge_update:
                self.emlps.append(
                    Sequential(
                        Linear(3 * self.hidden_channels, self.hidden_channels), 
                        ReLU(),
                        Linear(self.hidden_channels, self.hidden_channels)
                    )
                )
            self.convs.append(conv)
            if self.batch_norm:
                self.batch_norms.append(BatchNorm(self.hidden_channels))

        self.mlp = Sequential(
            Linear(self.hidden_channels*3, 50),
            ReLU(),
            Dropout(self.dropout),
            Linear(50, 25),
            ReLU(),
            Dropout(self.dropout),
            Linear(25, self.out_channels)
        )
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jk_mode:
            self.jk_linear.reset_parameters()
        if self.skip_connection:
            for nn in self.skip_proj:
                if isinstance(nn, Linear):
                    nn.reset_parameters()
        for layer in self.mlp:
            if isinstance(layer, Linear):
                layer.reset_parameters()
        for layer in self.emlps:
            if isinstance(layer, Linear):
                layer.reset_parameters()
        self.node_emb.reset_parameters()
        self.edge_emb.reset_parameters()
        for layer in self.batch_norms:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr, **kwargs):
        rev_edge_index = kwargs.pop("rev_edge_index", None)
        rev_edge_attr = kwargs.pop("rev_edge_attr", None)
        assert len(kwargs) == 0, "Unexpected arguments!"
        if rev_edge_index is None:
            return self.forward_default(x, edge_index, edge_attr)
        else:
            return self.forward_with_reverse_mp(x, edge_index, edge_attr, rev_edge_index, rev_edge_attr)

    def forward_with_reverse_mp(self, x, edge_index, edge_attr, rev_edge_index, rev_edge_attr):
        pass

    def forward_default(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        xs = []
        for i in range(self.num_layers):
            # x = F.dropout(x, p=self.dropout, training=self.training) Need full neighborhood information to capture local structural pattern
            if self.skip_connection:
                residual = self.skip_proj[i](x)
            conv_out = self.convs[i](x, edge_index)
            x = conv_out + residual if self.skip_connection else conv_out
            x = self.batch_norms[i](x) if self.batch_norm else x
            x = F.relu(x)
            if self.jk_mode != None:
                xs.append(x)

            if self.edge_update:
                if self.skip_connection:
                    residual = self.skip_proj[i](edge_attr)
                emlp_out = self.emlps[i](
                    torch.cat([x[src], x[dst], edge_attr], -1))
                edge_attr = emlp_out + residual if self.skip_connection else emlp_out

        # Dont know whether the relu is useful or not
        out = torch.cat([x[src].relu(), x[dst].relu(), edge_attr], -1)
        out = self.mlp(out)
        return out

       # Original (slow):
        # x = x[edge_index.T].reshape(-1, 2 * self.hidden_channels).relu()
        # x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        # return self.mlp(x)
        
