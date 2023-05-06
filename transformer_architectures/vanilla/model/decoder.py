import torch.nn as nn
from .utils import LayerNorm, SubLayerConnection, clones


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.features_dim)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, features_dim, attention, cross_attention, ff, dropout):
        super(DecoderLayer, self).__init__()
        self.attention = attention
        self.cross_attention = cross_attention
        self.ff = ff
        self.sublayers = clones(SubLayerConnection(features_dim=features_dim, dropout=dropout), 3)
        self.features_dim = features_dim

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.attention(x, x, x, tgt_mask))
        x = self.sublayers[1](
            x, lambda x: self.cross_attention(x, memory, memory, src_mask)
        )
        return self.sublayers[2](x, self.ff)
