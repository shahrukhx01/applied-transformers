import torch
import torch.nn as nn
import copy


class LayerNorm(nn.Module):
    def __init__(self, features_dim, epsilon=1e-9):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features_dim))
        self.beta = nn.Parameter(torch.zeros(features_dim))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * ((x - mean) / (std + self.epsilon)) + self.beta


class SubLayerConnection(nn.Module):
    def __init__(self, features_dim, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(features_dim=features_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


def clones(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
