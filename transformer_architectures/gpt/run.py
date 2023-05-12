import copy
import torch
import torch.nn as nn
import click
from transformer_architectures.vanilla.data.preprocess import build_vocab
from transformer_architectures.vanilla.inference.test_inference import inference_from_pretrained

from transformer_architectures.vanilla.model.attention import MultiHeadedAttention
from transformer_architectures.vanilla.model.embedding import PositionalEncoding, TokenEmbedding
from transformer_architectures.vanilla.model.decoder import Decoder, DecoderLayer
from transformer_architectures.vanilla.model.generator import Generator
from transformer_architectures.gpt.model.gpt import GPT
from transformer_architectures.vanilla.model.utils import PositionwiseFeedForward
from transformer_architectures.vanilla.train.train import train_worker


def make_model(
    vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = GPT(
        nn.Sequential(TokenEmbedding(vocab, d_model), c(position)),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model=d_model, vocab=vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model