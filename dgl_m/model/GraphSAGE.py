import os

os.environ['DGLBACKEND'] = "pytorch"

import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Callable, Optional, Tuple, Union
# from class_resolver.contrib.torch import activation_resolver
# from matplotlib import pyplot


class GraphSAGE_DGL(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        aggregator_type: str = "mean",
        activation: Union[str, Callable, None] = "relu",
        # activation_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[Callable, None] = None,
        jk: Optional[str] = None,
        config={},
        **kwargs,
    ):
        super().__init__()

        self.config = config
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU() if activation == "relu" else None
        self.norm = norm


        if num_layers > 1:
            self.convs.append(dglnn.pytorch.conv.SAGEConv(in_channels, hidden_channels, aggregator_type))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels

        for _ in range(num_layers - 2):
            self.convs.append(dglnn.pytorch.conv.SAGEConv(in_channels, hidden_channels, aggregator_type))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels

        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(dglnn.pytorch.conv.SAGEConv(in_channels, out_channels, aggregator_type))
        else:
            self.convs.append(dglnn.pytorch.conv.SAGEConv(in_channels, hidden_channels, aggregator_type))
        
        # if jk is not None:
        #     self.jk = dglnn.utils.JumpingKnowledge(jk, hidden_channels, num_layers)
        #     if jk == 'cat':
        #         in_channels = num_layers * hidden_channels
        #     else:
        #         in_channels = hidden_channels
        #     self.lin = nn.Linear(in_channels, self.out_channels)
        
    def reset_parameters(self):
        
        for layer in self.convs:
            layer.reset_parameters()
        
        # if hasattr(self, 'jk'):
        #     self.jk.reset_parameters()
        # if hasattr(self, 'lin'):
        #     self.lin.reset_parameters()


    def forward(
        self,
        mfgs,
        feat: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_weight: Tensor = None
    ):
        

        # hs: List[Tensor] = []
        
        for idx, layer in enumerate(self.convs):
            if idx == 0:
                h_dst = feat[:mfgs[0].num_dst_nodes()]
                h = layer(mfgs[0], (feat, h_dst), edge_weight)
            else:
                h_dst = h[:mfgs[idx].num_dst_nodes()]
                h = layer(mfgs[idx], (h, h_dst), edge_weight)

            if idx != self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)
            # if hasattr(self, 'jk'):
            #     hs.append(h)

        # h = self.jk(hs) if hasattr(self, 'jk') else h
        # h = self.lin(h) if hasattr(self, 'lin') else h

        return h

