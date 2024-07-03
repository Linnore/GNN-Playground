import os

os.environ['DGLBACKEND'] = "pytorch"

import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from typing import Union


class GAT_DGL_Custom(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_node_channels_per_head: int | list[int],
            num_layers: int,
            heads: int | list[int] = 8,
            output_heads: int = 1,
            dropout: float = 0.6,
            attention_dropout: float = 0.6,
            v2: bool = False,
            # jk: Union[str, None] = None,
            # skip_connection: bool = False,
            residual: bool = True,
            share_weights: bool = True,
            config={},
            *args,
            **kwargs
    ):
        
        super().__init__()

        self.config = config

        self.gat_layers = nn.ModuleList()
        
        if isinstance(hidden_node_channels_per_head, int):
            hidden_node_channels_per_head = [hidden_node_channels_per_head] * (num_layers-1)
        
        if isinstance(heads, int):
            heads = [heads] * (num_layers-1)

        if v2:
            self.conv = dglnn.GATv2Conv
            self.gat_layers.append(self.conv(in_feats=in_channels, out_feats=hidden_node_channels_per_head[0], 
                                            num_heads=heads[0], feat_drop=dropout, attn_drop=attention_dropout, activation=F.elu, share_weights=share_weights))

            for i in range(1, num_layers-1):
                self.gat_layers.append(self.conv(in_feats=hidden_node_channels_per_head[i-1]*heads[i-1], 
                                                out_feats=hidden_node_channels_per_head[i], num_heads=heads[i], 
                                                feat_drop=dropout, attn_drop=attention_dropout, activation=F.elu, residual=residual, share_weights=share_weights))

            self.gat_layers.append(self.conv(in_feats=hidden_node_channels_per_head[-1]*heads[-1], out_feats=out_channels, 
                                            num_heads=output_heads, feat_drop=dropout, attn_drop=attention_dropout, activation=None,
                                            residual=residual, share_weights=share_weights))
        
        else:
            self.conv = dglnn.GATConv
            self.gat_layers.append(self.conv(in_feats=in_channels, out_feats=hidden_node_channels_per_head[0], 
                                            num_heads=heads[0], feat_drop=dropout, attn_drop=attention_dropout, activation=F.elu))

            for i in range(1, num_layers-1):
                self.gat_layers.append(self.conv(in_feats=hidden_node_channels_per_head[i-1]*heads[i-1], 
                                                out_feats=hidden_node_channels_per_head[i], num_heads=heads[i], 
                                                feat_drop=dropout, attn_drop=attention_dropout, activation=F.elu, residual=residual))

            self.gat_layers.append(self.conv(in_feats=hidden_node_channels_per_head[-1]*heads[-1], out_feats=out_channels, 
                                            num_heads=output_heads, feat_drop=dropout, attn_drop=attention_dropout, activation=None,
                                            residual=residual))

        self.dropout = dropout
        self.num_layers = num_layers

    #     self.jk_mode = jk
    #     if not self.jk_mode in ['cat', None]:
    #         logger.exception(NotImplementedError("JK mode not implemented. Only support concat JK for now!"))

    #     if self.jk_mode != None:
    #         self.jk = dglnn.JumpingKnowledge(mode=jk)
    #         jk_in_channels = out_channels
    #         for i in range(len(heads)):
    #             jk_in_channels += heads[i] * hidden_node_channels_per_head[i]
    #         self.jk_linear = nn.Linear(jk_in_channels, out_channels)

    #     self.skip_connection = skip_connection
    #     if self.skip_connection:
    #         self.skip_proj = nn.ModuleList()
    #         self.skip_proj.append(self.get_skip_proj(in_channels, hidden_node_channels_per_head[0]*heads[0]))
    #         for i in range(1, num_layers-1):
    #             self.skip_proj.append(self.get_skip_proj(hidden_node_channels_per_head[i-1]*heads[i-1], hidden_node_channels_per_head[i]*heads[i]))

    #         self.skip_proj.append(self.get_skip_proj(hidden_node_channels_per_head[-1]*heads[-1], out_channels))

    # def get_skip_proj(self, in_channels, out_channels):
    #     if in_channels == out_channels:
    #         return nn.Identity()
    #     else:
    #         return nn.Linear(in_channels, out_channels)
        
    def reset_parameters(self):
        for conv in self.gat_layers:
            conv.reset_parameters()
        # if self.skip_connection:
        #     for sk in self.skip_proj:
        #         sk.reset_parameters()
    
    def forward(
            self,
            mfgs,
            feat,
    ):
        h = feat
        for i, layer in enumerate(self.gat_layers):
            if isinstance(mfgs, list):
                h = layer(mfgs[i], h)
            else:
                h = layer(mfgs, h)
            if i == self.num_layers-1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
        logger.debug(h.shape)
        return h
        