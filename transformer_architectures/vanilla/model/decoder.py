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
    def __init__(self, features_dim, attention, cross_attention, ff, dropout, 
                 decoder_only=False):
        super(DecoderLayer, self).__init__()
        self.attention = attention
        self.cross_attention = cross_attention
        self.ff = ff
        # set decoder only for GPT-like models
        self.decoder_only = decoder_only
        self.sublayers = clones(SubLayerConnection(features_dim=features_dim, 
                                            dropout=dropout), 2 if decoder_only else 3)
        self.features_dim = features_dim

    def forward(self, x, memory=None, src_mask=None, tgt_mask=None):
        x = self.sublayers[0](x, lambda x: self.attention(x, x, x, tgt_mask))
        # skip cross attention for GPT-like(decoder-only) models.
        if not self.decoder_only:
            x = self.sublayers[1](
                x, lambda x: self.cross_attention(x, memory, memory, src_mask)
            )
        return self.sublayers[-1](x, self.ff)
