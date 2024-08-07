import torch
import torch.nn.functional as F

from loguru import logger  # noqa
from typing import Literal

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

    def __init__(self,
                 in_channels,
                 hidden_channels: int | list[int],
                 num_layers: int,
                 out_channels: int,
                 deg,
                 aggregators=['mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 dropout: float = 0.6,
                 jk=None,
                 skip_connection: bool = False,
                 reverse_mp: bool = False,
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
        self.jk = jk
        self.skip_connection = skip_connection

        self.aggregators = aggregators
        self.scalars = scalers
        self.deg = deg

        self.init_layers()

        self.reverse_mp = reverse_mp
        if self.reverse_mp:
            self.init_layers_reverse_mp()

    def init_layers(self):
        if isinstance(self.hidden_channels, int):
            self.hidden_channels = [self.hidden_channels
                                    ] * (self.num_layers - 1)

        self.convs = ModuleList()
        self.convs.append(
            self.Conv(self.in_channels,
                      self.hidden_channels[0],
                      self.aggregators,
                      self.scalers,
                      deg=self.deg))
        for i in range(1, self.num_layers - 1):
            self.convs.append(
                self.Conv(self.hidden_channels[i - 1],
                          self.hidden_channels[i],
                          self.aggregators,
                          self.scalers,
                          deg=self.deg))
        self.convs.append(
            self.Conv(self.hidden_channels[-1],
                      self.out_channels,
                      self.aggregators,
                      self.scalers,
                      deg=self.deg))

        self.jk_mode = self.jk
        if self.jk_mode not in ["cat", None]:
            raise NotImplementedError(
                NotImplementedError(
                    "JK mode not implemented. Only support concat JK for now!")
            )
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(self.jk)
            jk_in_channels = self.out_channels
            for i in range(len(self.hidden_channels)):
                jk_in_channels += self.hidden_channels[i]
            self.jk_linear = Linear(jk_in_channels, self.out_channels)

        self.skip_connection = self.skip_connection
        if self.skip_connection:
            self.skip_proj = ModuleList()
            self.skip_proj.append(
                self.get_skip_proj(self.in_channels, self.hidden_channels[0]))
            for i in range(1, self.num_layers - 1):
                self.skip_proj.append(
                    self.get_skip_proj(self.hidden_channels[i - 1],
                                       self.hidden_channels[i]))
            self.skip_proj.append(
                self.get_skip_proj(self.hidden_channels[-1],
                                   self.out_channels))

    def init_layers_reverse_mp(self):
        if isinstance(self.hidden_channels, int):
            self.hidden_channels = [self.hidden_channels
                                    ] * (self.num_layers - 1)

        self.rev_convs = ModuleList()
        self.rev_convs.append(
            self.Conv(self.in_channels,
                      self.hidden_channels[0],
                      self.aggregators,
                      self.scalers,
                      deg=self.deg))
        for i in range(1, self.num_layers - 1):
            self.rev_convs.append(
                self.Conv(self.hidden_channels[i - 1],
                          self.hidden_channels[i],
                          self.aggregators,
                          self.scalers,
                          deg=self.deg))
        self.rev_convs.append(
            self.Conv(self.hidden_channels[-1],
                      self.out_channels,
                      self.aggregators,
                      self.scalers,
                      deg=self.deg))

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

            if i != self.num_layers - 1:
                x = F.elu(x)

            if self.jk_mode is not None:
                xs.append(x)

        if self.jk_mode is not None:
            x = self.jk(xs)
            x = self.jk_linear(x)
        return x


class PNAe_layer_mix(PNA_Custom):
    # Adjusted model architecture from
    # https://github.com/IBM/Multi-GNN/blob/252b0252afca109d1d216c411c59ff70753b25fc/models.py#L7
    def __init__(self,
                 edge_update: bool = False,
                 edge_dim=None,
                 batch_norm=True,
                 layer_mix: Literal["None", "Mean", "Sum", "Max",
                                    "Cat"] = "Mean",
                 *args,
                 **kwargs):
        self.batch_norm = batch_norm
        self.edge_update = edge_update
        self.edge_dim = edge_dim
        self.layer_mix = layer_mix
        self.readout = kwargs.get('readout', None)

        super().__init__(
            *args,
            **kwargs,
        )

    def init_layers(self):
        self.node_emb = Linear(self.in_channels, self.hidden_channels)
        self.edge_emb = Linear(self.edge_dim, self.hidden_channels)

        self.convs = ModuleList()
        self.emlps = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(self.num_layers):
            conv = self.Conv(in_channels=self.hidden_channels,
                             out_channels=self.hidden_channels,
                             aggregators=self.aggregators,
                             scalers=self.scalars,
                             deg=self.deg,
                             edge_dim=self.hidden_channels,
                             towers=5,
                             pre_layers=1,
                             post_layers=1,
                             divide_input=False)
            if self.edge_update:
                self.emlps.append(
                    Sequential(
                        Linear(3 * self.hidden_channels, self.hidden_channels),
                        ReLU(),
                        Linear(self.hidden_channels, self.hidden_channels)))
            self.convs.append(conv)
            if self.batch_norm:
                self.batch_norms.append(BatchNorm(self.hidden_channels))

        if self.readout == "edge":
            readout_in_channels = self.hidden_channels * 3
        elif self.readout == "node":
            readout_in_channels = self.hidden_channels

        self.mlp = Sequential(Linear(readout_in_channels, 50), ReLU(),
                              Dropout(self.dropout), Linear(50, 25), ReLU(),
                              Dropout(self.dropout),
                              Linear(25, self.out_channels))

        self.jk_mode = self.jk
        if self.jk_mode not in ["cat", None]:
            raise NotImplementedError(
                NotImplementedError(
                    "JK mode not implemented. Only support concat JK for now!")
            )
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(self.jk)
            jk_in_channels = self.hidden_channels * (self.num_layers + 1)
            self.jk_linear = Linear(jk_in_channels, self.out_channels)

        self.skip_connection = self.skip_connection
        if self.skip_connection:
            self.skip_proj = ModuleList()
            for i in range(1, self.num_layers):
                self.skip_proj.append(
                    self.get_skip_proj(self.hidden_channels,
                                       self.hidden_channels))

    def init_layers_reverse_mp(self):
        self.rev_edge_emb = Linear(self.edge_dim, self.hidden_channels)

        self.rev_convs = ModuleList()
        self.rev_emlps = ModuleList()
        for _ in range(self.num_layers):
            conv = self.Conv(in_channels=self.hidden_channels,
                             out_channels=self.hidden_channels,
                             aggregators=self.aggregators,
                             scalers=self.scalars,
                             deg=self.deg,
                             edge_dim=self.hidden_channels,
                             towers=5,
                             pre_layers=1,
                             post_layers=1,
                             divide_input=False)
            if self.edge_update:
                self.rev_emlps.append(
                    Sequential(
                        Linear(3 * self.hidden_channels, self.hidden_channels),
                        ReLU(),
                        Linear(self.hidden_channels, self.hidden_channels)))
            self.rev_convs.append(conv)

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

    def get_mixture(self, fx, rx):
        match self.layer_mix.lower():
            case "none":
                raise NotImplementedError
            case "mean":
                return (fx + rx) / 2
            case "sum":
                return fx + rx
            case "cat":
                return torch.concatenate((fx, rx))
            case "max":
                return torch.max(fx, rx)
            case _:
                raise NotImplementedError

    def forward(self, x, edge_index, edge_attr, **kwargs):
        rev_edge_index = kwargs.pop("rev_edge_index", None)
        rev_edge_attr = kwargs.pop("rev_edge_attr", None)
        assert len(kwargs) == 0, "Unexpected arguments!"

        if rev_edge_index is None:
            return self.forward_default(x, edge_index, edge_attr)
        else:
            return self.forward_with_reverse_mp(x, edge_index, edge_attr,
                                                rev_edge_index, rev_edge_attr)

    def forward_with_reverse_mp(self, x, edge_index, edge_attr, rev_edge_index,
                                rev_edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        rev_edge_attr = self.rev_edge_emb(rev_edge_attr)

        xs = [x]
        for i in range(self.num_layers):
            if self.skip_connection:
                residual = Identity(x)
            # non-reverse
            fx = self.convs[i](x, edge_index, edge_attr)

            # reverse
            rx = self.rev_convs[i](x, rev_edge_index, rev_edge_attr)

            # Mix
            mix_out = self.get_mixture(fx, rx)
            # mix_out = (fx + rx) / 2 # TODO: Try different mixture method

            x = mix_out + residual if self.skip_connection else mix_out
            x = self.batch_norms[i](x) if self.batch_norm else x
            x = F.relu(x)
            if self.jk_mode is not None:
                xs.append(x)

            if self.edge_update:
                if self.skip_connection:
                    residual = self.skip_proj[i](edge_attr)
                emlp_out = self.emlps[i](torch.cat([x[src], x[dst], edge_attr],
                                                   -1))
                if self.skip_connection:
                    edge_attr = emlp_out + residual
                else:
                    edge_attr = emlp_out

        if self.jk_mode is not None:
            x = self.jk_linear(self.jk(xs))

        if self.readout == "edge":
            # Dont know whether the relu is useful or not
            out = torch.cat([x[src].relu(), x[dst].relu(), edge_attr], -1)
            out = self.mlp(out)
            return out
        elif self.readout == "node":
            out = self.mlp(x)
            return out

    def forward_default(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        xs = [x]
        for i in range(self.num_layers):
            # x = F.dropout(x, p=self.dropout, training=self.training)
            if self.skip_connection:
                residual = self.skip_proj[i](x)
            conv_out = self.convs[i](x, edge_index, edge_attr)
            x = conv_out + residual if self.skip_connection else conv_out
            x = self.batch_norms[i](x) if self.batch_norm else x

            if i != self.num_layers - 1:
                x = F.relu(x)

            if self.jk_mode is not None:
                xs.append(x)

            if self.edge_update:
                if self.skip_connection:
                    residual = self.skip_proj[i](edge_attr)
                emlp_out = self.emlps[i](torch.cat([x[src], x[dst], edge_attr],
                                                   -1))
                if self.skip_connection:
                    edge_attr = emlp_out + residual
                else:
                    edge_attr = emlp_out

        if self.jk_mode is not None:
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


class PNAe(torch.nn.Module):

    def __init__(self,
                 edge_update: bool = False,
                 edge_dim=None,
                 batch_norm=True,
                 layer_mix: Literal["None", "Mean", "Sum", "Max",
                                    "Cat"] = "Mean",
                 model_mix: Literal["Mean", "Sum", "Max"] = "Mean",
                 *args,
                 **kwargs):

        super().__init__()

        self.reverse_mp = kwargs.get("reverse_mp", False)
        self.layer_mix = layer_mix
        self.model_mix = model_mix
        self.config = kwargs.get("config", {})

        if self.reverse_mp and self.layer_mix.lower() == "none":
            kwargs["reverse_mp"] = False
            self.org_model = PNAe_layer_mix(edge_update=edge_update,
                                            edge_dim=edge_dim,
                                            batch_norm=batch_norm,
                                            layer_mix=layer_mix,
                                            *args,
                                            **kwargs)

            self.rev_model = PNAe_layer_mix(edge_update=edge_update,
                                            edge_dim=edge_dim,
                                            batch_norm=batch_norm,
                                            layer_mix=layer_mix,
                                            *args,
                                            **kwargs)

        else:
            self.model = PNAe_layer_mix(edge_update=edge_update,
                                        edge_dim=edge_dim,
                                        batch_norm=batch_norm,
                                        layer_mix=layer_mix,
                                        *args,
                                        **kwargs)

    def forward(self, x, edge_index, edge_attr, **kwargs):
        if self.reverse_mp and self.layer_mix.lower() == "none":
            rev_edge_index = kwargs.pop("rev_edge_index", None)
            rev_edge_attr = kwargs.pop("rev_edge_attr", None)
            assert len(kwargs) == 0, "Unexpected arguments!"

            org_out = self.org_model(x, edge_index, edge_attr)
            rev_out = self.rev_model(x, rev_edge_index, rev_edge_attr)
            return self.get_model_mixture(org_out, rev_out)
        else:
            return self.model(x, edge_index, edge_attr, **kwargs)

    def reset_parameters(self):
        if self.reverse_mp and self.layer_mix.lower() == "none":
            self.org_model.reset_parameters()
            self.rev_model.reset_parameters()
        else:
            self.model.reset_parameters()

    def get_model_mixture(self, org_out, rev_out):
        match self.model_mix.lower():
            case "mean":
                return (org_out + rev_out) / 2
            case "sum":
                return org_out + rev_out
            case "max":
                return torch.max(org_out, rev_out)
            case _:
                raise NotImplementedError
