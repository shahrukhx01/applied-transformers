import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, src_embedding, tgt_embedding, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(self.src_embedding(src), src_mask)
        return self.decoder(self.tgt_embedding(tgt), memory, src_mask, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embedding(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embedding(tgt), memory, src_mask, tgt_mask)