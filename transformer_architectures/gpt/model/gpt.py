import torch.nn as nn
from torch.nn import functional as F


class GPT(nn.Module):
    def __init__(self, embedding, decoder, generator, d_model, vocab_size):
        super(GPT, self).__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.generator = generator
    
    def forward(self, x, mask, targets=None):
        x = self.decoder(self.embedding(x), memory=None, src_mask=None, 
                            tgt_mask=mask)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    



