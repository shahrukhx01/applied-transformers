import torch.nn as nn
import torch
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


"""
More intuitive, however, not the most optimal
Better for understanding the intuition:

PE_(pos,2i) = sin(pos/10000^(2i/d_model))
PE_(pos,2i+1) = cos(pos/10000^(2i/d_model))

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_length, d_model):
        self.max_sequence_length = max_seq_length
        self.d_model = d_model

    def forward(self, x):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(
            self.max_sequence_length, 1
        )
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
"""


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class TransformerEmbedding(nn.Module):
    def __init__(self, token_embedding, pe_embedding):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = token_embedding
        self.pe_embedding = pe_embedding
    
    def forward(self, x):
        return self.pe_embedding(self.token_embedding(x))