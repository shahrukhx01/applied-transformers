from torch.utils.data.distributed import DistributedSampler
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from transformer_architectures.vanilla.data.batch import collate_batch, collate_batch_decoder_only
import pandas as pd

from transformer_architectures.vanilla.data.dataset import TransformerDataset

def create_dataloaders(
    train_dataset_path,
    val_dataset_path,
    src_column, 
    tgt_column,
    device,
    vocab_src,
    vocab_tgt,
    batch_size=12000,
    max_padding=128,
    is_distributed=False,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize(text):
        return str(text).split()

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize,
            tokenize,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter = TransformerDataset(train_dataset_path, src_column=src_column, 
                                    tgt_column=tgt_column)
    valid_iter = TransformerDataset(val_dataset_path, src_column=src_column, 
                                    tgt_column=tgt_column)

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader

def create_dataloaders_decoder_only(
    train_dataset_path,
    val_dataset_path,
    src_column, 
    tgt_column,
    device,
    vocab,
    tokenization_fn,
    batch_size=12_000,
    max_padding=128,
    is_distributed=False,
):

    def collate_fn(batch):
        return collate_batch_decoder_only(
            batch,
            tokenization_fn,
            vocab,
            device,
            max_padding=max_padding,
            pad_id=vocab.get_stoi()["<blank>"],
        )

    train_iter = TransformerDataset(train_dataset_path, src_column=src_column, 
                                    tgt_column=tgt_column)
    valid_iter = TransformerDataset(val_dataset_path, src_column=src_column, 
                                    tgt_column=tgt_column)

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader