import copy
import torch
import torch.nn as nn
import click
from transformer_architectures.vanilla.data.dataloader import create_dataloaders_decoder_only
from transformer_architectures.vanilla.data.preprocess import build_vocab, char_level_tokenizer
from transformer_architectures.gpt.inference.inference import inference_from_pretrained
from transformer_architectures.vanilla.model.attention import MultiHeadedAttention
from transformer_architectures.vanilla.model.embedding import PositionalEncoding, TokenEmbedding
from transformer_architectures.vanilla.model.decoder import Decoder, DecoderLayer
from transformer_architectures.vanilla.model.generator import Generator
from transformer_architectures.gpt.model.gpt import GPT
from transformer_architectures.vanilla.model.utils import PositionwiseFeedForward
from transformer_architectures.gpt.train.train import train_worker
from transformer_architectures.vanilla.data.preprocess import tokenizer_fn_map
from torch.nn import functional as F


def make_model(
    vocab_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = GPT(
        nn.Sequential(TokenEmbedding(vocab_size, d_model), c(position)),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout, decoder_only=True), N),
        Generator(d_model=d_model, vocab=vocab_size),
        d_model=d_model, 
        vocab_size=vocab_size,
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
              help="Total number of GPT layers (default: 6)")
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
@click.option("--load_pretrained", "-p",
              default=None,
              help="Path to pre-trained checkpoint, (default None)")
@click.option("--tokenizer", "-t",
              default="word",
              type=click.Choice(["word", "char"]),
              help="Type of tokenizer, types include `word`, `char`, (default `word`)")
def gpt_run(num_layers, d_model, d_ff, num_heads, dropout, 
                    train_path, valid_path, load_pretrained, tokenizer):
    config = {
        "batch_size": 1,
        "distributed": False,
        "num_epochs": 1,
        "accum_iter": 1,
        "base_lr": 1.0,
        "max_padding": 410,
        "warmup": 10,
        "file_prefix": "transformer_test",
    }    
    vocab = build_vocab(
                file_path=train_path,
                tokenizer_type=tokenizer,
                src_column='source',
                tgt_column='target',
            )     
    gpt_model = make_model(
    vocab_size=len(vocab), N=num_layers, d_model=d_model, 
    d_ff=d_ff, h=num_heads, dropout=dropout 
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### KARPATHY GPT
    train_dataloader, valid_dataloader = create_dataloaders_decoder_only(
        train_path,
        valid_path,
        src_column='source',
        tgt_column='target',
        device=device,
        vocab=vocab,
        tokenization_fn=tokenizer_fn_map[tokenizer],
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
    )
    learning_rate = 3e-4
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=learning_rate)
    gpt_model.to(device)
    file_path = "%sfinal.pt" % config["file_prefix"]
    gpt_model.load_state_dict(
        torch.load(file_path, map_location=device)
    )
    for iter in range(2000):
        if iter > 0:
            file_path = "%sfinal.pt" % config["file_prefix"]
            torch.save(gpt_model.state_dict(), file_path)
        # sample a batch of data
        for idx, batch in enumerate(train_dataloader):
            # evaluate the loss
            logits, loss = gpt_model(x=batch.tgt.to(device), mask=batch.tgt_mask.to(device), targets=batch.tgt_y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if idx%1000 == 0:
                print(f"Train loss for step {idx+1} in epoch {iter+1}: {loss.item()}")
                prob =F.softmax(logits, dim=-1)
                text = "".join([vocab.get_itos()[char.item()] for char in batch.tgt[0].detach().cpu() if char.item()!=2])
                print(text)
                pred = []
                for idx, pchar in enumerate(prob.detach().cpu()):
                     pred_char = vocab.get_itos()[torch.argmax(pchar).item()]
                     if pred_char != '<blank>':
                         pred.append(pred_char)
                print("".join(pred))

if __name__ == "__main__":
    gpt_run()
