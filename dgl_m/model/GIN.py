import os

os.environ['DGLBACKEND'] = "pytorch"

import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from typing import Union


class MLP(nn.Module):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_lin_layers: int,
            mlp_bias: bool = True,
            mlp_dropout: float = 0.3,
    ):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=mlp_bias))
        for _ in range(1, num_lin_layers-1):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=mlp_bias))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=mlp_bias))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))
        self.mlp_dropout = mlp_dropout

    def forward(self, x):
        h = x
        for _, layer in enumerate(self.linears[:-1]):
            h = layer(h)
            h = self.batch_norm(h)
            h = F.relu(h)
            h = F.dropout(h, self.mlp_dropout)
        return self.linears[-1](h)
    
    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
        self.batch_norm.reset_parameters()

class GIN_DGL_Custom(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_lin_layers: int,
            num_gcn_layers: int,
            aggregator_type: str = 'sum',
            init_eps: float = 0,
            learn_eps: bool = False,
            mlp_activation: Union[str, None] = None,
            gin_activation: Union[str, None] = None,
            dropout: float = 0.6,
            mlp_dropout: float = 0.3,
            mlp_bias: bool = True,
            config={},
            # pooling: str = 'sum',
    ):
        super().__init__()

        self.config = config

        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        mlp_activation = nn.ReLU() if mlp_activation == "relu" else None
        self.gin_activation = nn.ReLU() if gin_activation == "relu" else None

        for layer in range(num_gcn_layers):
            if layer == 0:
                self.mlp = MLP(input_dim, hidden_dim, hidden_dim, num_lin_layers, mlp_bias, mlp_dropout)
            else:
                self.mlp = MLP(hidden_dim, hidden_dim, hidden_dim, num_lin_layers, mlp_bias, mlp_dropout)
            self.ginlayers.append(dglnn.GINConv(self.mlp, aggregator_type=aggregator_type, init_eps=init_eps, 
                                                learn_eps=learn_eps, activation=mlp_activation))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # self.linear_prediction = nn.ModuleList()
        # for layer in range(num_gcn_layers):
        #     if layer == 0:
        #         self.linear_prediction.append(nn.Linear(input_dim, output_dim))
        #     else:
        #         self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))

        # self.dropout = nn.Dropout(p=dropout)
        # if pooling == 'avg':
        #     self.pool = (dglnn.AvgPooling())
        # elif pooling == 'max':
        #     self.pool = (dglnn.MaxPooling())
        # else:
        #     self.pool = (dglnn.SumPooling())

        self.dropout = dropout
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        
    def forward(self, mfgs, h):
        # hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            
            if isinstance(mfgs, list):
                h = layer(mfgs[i], h)
            else:
                h = layer(mfgs, h)
            
            h = self.batch_norms[i](h)
            h = self.gin_activation(h)
            # hidden_rep.append(h)
        h = self.gin_activation(self.linear1(h))
        h = F.dropout(h, self.dropout)
        h = self.linear2(h)
        return h

        # score_over_layer = 0
        # for i, h in enumerate(hidden_rep):
        #     if isinstance(mfgs, list):
        #         pooled_h = self.pool(mfgs[i], h)
        #     else:
        #         pooled_h = self.pool(mfgs, h)
        #     score_over_layer += self.dropout(self.linear_prediction[i](pooled_h))
        # return score_over_layer
    
    def reset_parameters(self):
        # for gin in self.ginlayers:
        #     print(gin)
        #     gin.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        self.mlp.reset_parameters()
        # for lp in self.linear_prediction:
        #     lp.reset_parameters()
