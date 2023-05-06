import torch.nn as nn
from .utils import LayerNorm, SubLayerConnection, clones


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.features_dim)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, features_dim, attention, ff, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.ff = ff
        self.sublayers = clones(SubLayerConnection(features_dim=features_dim, dropout=dropout), 2)
        self.features_dim = features_dim

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.attention(x, x, x, mask))
        return self.sublayers[1](x, self.ff)
