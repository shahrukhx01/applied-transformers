import torch.nn as nn
from .utils import LayerNorm, SubLayerConnection, clones


class Encoder(nn.Module):
    def __init__(self, layers, d_model):
        self.layers = layers
        self.norm = LayerNorm(d_model=d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, attention, ff, dropout, d_model):
        self.attention = attention
        self.ff = ff
        self.sublayers = clones(SubLayerConnection(d_model=d_model, dropout=dropout), 2)

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.attention(x, x, x, mask))
        return self.sublayers[1](x, self.ff)
