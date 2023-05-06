import copy
import torch.nn as nn
import click
from transformer_architectures.vanilla.data.preprocess import build_vocab

from transformer_architectures.vanilla.model.attention import MultiHeadedAttention
from transformer_architectures.vanilla.model.embedding import PositionalEncoding, TokenEmbedding
from transformer_architectures.vanilla.model.encoder import Encoder, EncoderLayer
from transformer_architectures.vanilla.model.decoder import Decoder, DecoderLayer
from transformer_architectures.vanilla.model.generator import Generator
from transformer_architectures.vanilla.model.transformer import Transformer
from transformer_architectures.vanilla.model.utils import PositionwiseFeedForward
from transformer_architectures.vanilla.train.train import train_worker


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
@click.option("--train_path", "-t",
              help="Train dataset path")
@click.option("--valid_path", "-v",
              help="Validation dataset path")
def transformer_run(num_layers, d_model, d_ff, num_heads, dropout, 
                    train_path, valid_path):
    config = {
        "batch_size": 8,
        "distributed": False,
        "num_epochs": 1,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 10,
        "file_prefix": "transformer_test",
    }    
    vocab = build_vocab(file_path=train_path,
                src_column='source',
                tgt_column='target',) 
    transformer_model = make_model(
    src_vocab=len(vocab), tgt_vocab=len(vocab), N=num_layers, d_model=d_model, 
    d_ff=d_ff, h=num_heads, dropout=dropout 
    )
    print(len(vocab))   
    train_worker(
    train_dataset_path=train_path,
    validation_dataset_path=valid_path,
    src_column='source',
    tgt_column='target',
    vocab_src=vocab,
    vocab_tgt=vocab,
    model=transformer_model,
    config=config,
    d_model = 512,
    is_distributed=False
)

if __name__ == "__main__":
    transformer_run()
