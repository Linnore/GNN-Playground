import torch
import torch.nn.functional as F

from loguru import logger

from torch.nn import Linear, Identity, ModuleList, Sequential, ReLU, Dropout

from torch_geometric.nn.models import GAT as GAT_Base
from torch_geometric.nn.models import JumpingKnowledge
from torch_geometric.nn import GATConv, GATv2Conv, BatchNorm


class GAT_PyG(GAT_Base):
    def __init__(self, config={}, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.config = config


class GAT_Custom(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels_per_head: int | list[int],
                 num_layers: int, out_channels: int,
                 heads: int | list[int] = 8,
                 output_heads: int = 1,
                 v2: bool = False,
                 dropout: float = 0.6,
                 jk=None,
                 skip_connection: bool = False,
                 config={},
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.config = config

        if v2:
            self.Conv = GATv2Conv
        else:
            self.Conv = GATConv

        self.in_channels = in_channels
        self.hidden_channels_per_head = hidden_channels_per_head
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        if type(hidden_channels_per_head) == int:
            hidden_channels_per_head = [
                hidden_channels_per_head] * (num_layers-1)

        if type(heads) == int:
            heads = [heads] * (num_layers-1)

        self.convs = ModuleList()
        self.convs.append(
            self.Conv(in_channels,
                 hidden_channels_per_head[0], heads[0], dropout=dropout)
        )
        for i in range(1, num_layers-1):
            self.convs.append(
                self.Conv(hidden_channels_per_head[i-1]*heads[i-1],
                     hidden_channels_per_head[i], heads[i], dropout=dropout)
            )
        self.convs.append(self.Conv(
            hidden_channels_per_head[-1] * heads[-1], out_channels, output_heads, concat=False, dropout=dropout))

        self.jk_mode = jk
        if not self.jk_mode in ["cat", None]:
            raise NotImplementedError(NotImplementedError(
                "JK mode not implemented. Only support concat JK for now!"))
        if self.jk_mode != None:
            self.jk = JumpingKnowledge(jk)
            jk_in_channels = out_channels
            for i in range(len(heads)):
                jk_in_channels += heads[i] * hidden_channels_per_head[i]
            self.jk_linear = Linear(jk_in_channels, out_channels)

        self.skip_connection = skip_connection
        if self.skip_connection:
            self.skip_proj = ModuleList()
            self.skip_proj.append(
                self.get_skip_proj(
                    in_channels, hidden_channels_per_head[0]*heads[0])
            )
            for i in range(1, num_layers-1):
                self.skip_proj.append(
                    self.get_skip_proj(
                        hidden_channels_per_head[i-1]*heads[i-1],
                        hidden_channels_per_head[i]*heads[i]
                    )
                )
            self.skip_proj.append(
                self.get_skip_proj(
                    hidden_channels_per_head[-1]*heads[-1], out_channels)
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
        rev_edge_index = kwargs.pop("rev_edge_index", None)
        assert len(kwargs) == 0, "Unexpected arguments!"
        if rev_edge_index is None:
            return self.forward_default(x, edge_index)
        else:
            return self.forward_with_reverse_mp(x, edge_index, rev_edge_index)

    def forward_with_reverse_mp(self, x, edge_index, rev_edge_index):
        pass

    def forward_default(self, x, edge_index):
        xs = []
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip_connection:
                residual = self.skip_proj[i](x)
            conv_out = self.convs[i](x, edge_index)
            x = conv_out + residual if self.skip_connection else conv_out
            
            if i != self.num_layers-1:
                x = F.elu(x)
            
            if self.jk_mode != None:
                xs.append(x)

        if self.jk_mode != None:
            x = self.jk(xs)
            x = self.jk_linear(x)
        return x
    
    
class GATe(GAT_Custom):
    # Adjusted model architecture from https://github.com/IBM/Multi-GNN/blob/252b0252afca109d1d216c411c59ff70753b25fc/models.py#L7
    def __init__(self,
                 in_channels: int,
                 hidden_channels_per_head: int,
                 out_channels: int,
                 heads: int,
                 edge_update: bool = False,
                 edge_dim=None,
                 batch_norm=True,
                 *args,
                 **kwargs
                 ):
        
        self.hidden_channels = hidden_channels_per_head * heads
        
        super().__init__(
            in_channels=self.hidden_channels,
            hidden_channels_per_head=hidden_channels_per_head,
            heads=heads,
            out_channels=self.hidden_channels,
            *args,
            **kwargs,
        )

        self.batch_norm = batch_norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_update = edge_update
        self.readout = kwargs.get('readout', None)

        self.node_emb = Linear(self.in_channels, self.hidden_channels)
        self.edge_emb = Linear(edge_dim, self.hidden_channels)

        self.convs = ModuleList()
        self.emlps = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(self.num_layers):
            conv = self.Conv(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels, 
                heads=heads, 
                concat=True,
                dropout=self.dropout,
                add_self_loops=True,
                edge_dim=self.hidden_channels
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

        if self.readout == "edge":
            readout_in_channels = self.hidden_channels*3
        elif self.readout == "node":
            readout_in_channels = self.hidden_channels
            
        self.mlp = Sequential(
            Linear(readout_in_channels, 50),
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

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        xs = []
        for i in range(self.num_layers):
            # x = F.dropout(x, p=self.dropout, training=self.training) Need full neighborhood information to capture local structural pattern
            if self.skip_connection:
                residual = self.skip_proj[i](x)
            conv_out = self.convs[i](x, edge_index, edge_attr)
            x = conv_out + residual if self.skip_connection else conv_out
            x = self.batch_norms[i](x) if self.batch_norm else x
            
            if i != self.num_layers-1:
                x = F.relu(x)
                
            if self.edge_update:
                if self.skip_connection:
                    residual = self.skip_proj[i](edge_attr)
                emlp_out = self.emlps[i](
                    torch.cat([x[src], x[dst], edge_attr], -1))
                edge_attr = emlp_out + residual if self.skip_connection else emlp_out
                
        if self.jk_mode != None:
            x = self.jk_linear(self.jk(xs))

        if self.readout == "edge":
        # Dont know whether the relu is useful or not
            out = torch.cat([x[src].relu(), x[dst].relu(), edge_attr], -1)
            out = self.mlp(out)
            return out
        elif self.readout == "node":
            out = self.mlp(x)
            return out

        # Original (slow):
        # x = x[edge_index.T].reshape(-1, 2 * self.hidden_channels).relu()
        # x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        # return self.mlp(x)
