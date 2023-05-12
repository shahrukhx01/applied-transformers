import time
import torch
from transformer_architectures.vanilla.data.dataloader import  create_dataloaders_decoder_only
from transformer_architectures.vanilla.data.preprocess import tokenizer_fn_map
from torch.optim.lr_scheduler import LambdaLR

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

def run_epoch(
    data_iter,
    model,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_loss = 0
    total_tokens = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        _, loss = model.forward(
            x=batch.tgt, mask=batch.tgt_mask, targets=batch.tgt_y
        )
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Learning Rate: %6.1e"
                )
                % (i, n_accum, total_loss / batch.ntokens, lr)
            )
            start = time.time()
        del loss
    return total_loss / total_tokens, train_state


def train_worker(
    train_dataset_path,
    validation_dataset_path,
    src_column,
    tgt_column,
    vocab,
    model,
    config,
    tokenizer,
    d_model = 512,
    is_distributed=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Train worker process using GPU: {device} for training", flush=True)


    pad_idx = vocab["<blank>"]
    model.to(device)
    module = model
    is_main_process = True



    train_dataloader, valid_dataloader = create_dataloaders_decoder_only(
        train_dataset_path,
        validation_dataset_path,
        src_column=src_column,
        tgt_column=tgt_column,
        device=device,
        vocab=vocab,
        tokenization_fn=tokenizer_fn_map[tokenizer],
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[Device {device}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            train_dataloader,
            model,
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[Device {device}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            valid_dataloader,
            model,
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)