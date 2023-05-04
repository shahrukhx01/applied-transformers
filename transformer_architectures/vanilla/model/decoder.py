import torch.nn as nn
from .utils import LayerNorm, SubLayerConnection, clones


class Decoder(nn.Module):
    def __init__(self, layers, d_model):
        self.layers = layers
        self.norm = LayerNorm(d_model=d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, attention, cross_attention, ff, dropout, d_model):
        self.attention = attention
        self.cross_attention = cross_attention
        self.ff = ff
        self.sublayers = clones(SubLayerConnection(d_model=d_model, dropout=dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.attention(x, x, x, tgt_mask))
        x = self.sublayers[1](
            x, lambda x: self.cross_attention(x, memory, memory, src_mask)
        )
        return self.sublayers[2](x, self.ff)
