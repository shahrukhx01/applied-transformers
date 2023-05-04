import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, src_embedding, tgt_embedding, encoder, decoder):
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(self.src_embedding(src), src_mask)
        return self.decoder(self.tgt_embedding(tgt), memory, src_mask, tgt_mask)
