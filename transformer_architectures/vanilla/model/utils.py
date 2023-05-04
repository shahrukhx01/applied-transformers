import torch
import torch.nn as nn
import copy


class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-9):
        self.gamma = nn.Parameter(nn.ones(d_model))
        self.beta = nn.Parameter(nn.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * ((x - mean) / (std + self.epsilon)) + self.beta


class SubLayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        self.norm = LayerNorm(d_model=d_model)
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


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def clones(layer, N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
