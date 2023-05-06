import copy
import torch.nn as nn
import click

from transformer_architectures.vanilla.model.attention import MultiHeadedAttention
from transformer_architectures.vanilla.model.embedding import PositionalEncoding, TokenEmbedding
from transformer_architectures.vanilla.model.encoder import Encoder, EncoderLayer
from transformer_architectures.vanilla.model.decoder import Decoder, DecoderLayer
from transformer_architectures.vanilla.model.generator import Generator
from transformer_architectures.vanilla.model.transformer import Transformer
from transformer_architectures.vanilla.model.utils import PositionwiseFeedForward


def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        nn.Sequential(TokenEmbedding(src_vocab, d_model), c(position)),
        nn.Sequential(TokenEmbedding(tgt_vocab, d_model), c(position)),
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model=d_model, vocab=tgt_vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

@click.command()
@click.option("--num_layers", "-n",
              default=6,
              help="Total number of transformer layers (default: 6)")
@click.option("--d_model", "-m",
              default=512,
              help="Hidden dimension of the encoder/decoder (default: 512)")
@click.option("--d_ff", "-f",
              default=2048,
              help="Dimension of the Feedforward layer (default: 2048)")
@click.option("--num_heads", "-h",
              default=8,
              help="Number of heads in multi-head attention (default: 8)")
@click.option("--dropout", "-d",
              default=0.1,
              help="Dropout probability (default: 0.1)")
def transformer_run(num_layers, d_model, d_ff, num_heads, dropout):
    transformer_model = make_model(
    src_vocab=10, tgt_vocab=10, N=num_layers, d_model=d_model, 
    d_ff=d_ff, h=num_heads, dropout=dropout 
    )
    print(transformer_model)    

if __name__ == "__main__":
    transformer_run()
