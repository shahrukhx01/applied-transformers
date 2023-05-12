import torch.nn as nn


class GPT(nn.Module):
    def __init__(self, embedding, decoder, generator):
        super(GPT, self).__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.generator = generator
    
    def forward(self, x, mask):
        memory = None
        return self.decoder(self.embedding(x), memory, src_mask=None, tgt_mask=mask)
    



