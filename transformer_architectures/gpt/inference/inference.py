import torch
from transformer_architectures.vanilla.data.batch import Batch
from torch.nn.functional import pad
from torch.nn import functional as F

def inference_from_pretrained(model, example, vocab, src_pipeline, 
                              device, max_padding = 409, pad_id=2):
    model.eval()
    model.to(device)
    bs_id = torch.tensor([0], device=device)  # <s> token id
    generation = ""
    for i in range(100):
        src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    vocab(src_pipeline(example, generation)),
                    dtype=torch.int64,
                    device=device,
                ),
            ],
            0,
        ).unsqueeze(0)
        src = pad(
                src,
                (
                    0,
                    max_padding - src.shape[1],
                ),
                value=pad_id,
            )
        src_mask = Batch.make_std_mask(src, pad=pad_id)
        src.to(device)
        src_mask.to(device)
        logits, _ = model.forward(
            src, src_mask
        )
        prob =F.softmax(logits[0], dim=-1)
        next_char = "".join([vocab.get_itos()[torch.argmax(pchar).item()] for pchar in prob if torch.argmax(pchar).item()!=2 ][len(example)+i+1:])
        if next_char == '</s>':
            break
        generation += next_char
    print(f"shahrukh: {generation}")


