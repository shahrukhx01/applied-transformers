import torch
from transformer_architectures.vanilla.data.batch import Batch, subsequent_mask
from torch.nn.functional import pad
from torch.nn.functional import log_softmax
from torch.nn import functional as F

def inference_from_pretrained(model, example, vocab, src_pipeline, 
                              device='cpu', max_padding = 409, pad_id=2):
    
    model.eval()
    model.to(device)
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    # ans = "Ok"
    generation = ""
    for i in range(10):
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
        # print([vocab.get_itos()[i.detach().cpu().item()] for i,mask in zip(src[0], src_mask[0][400]) if mask])
        # memory =None
        # print(src_mask)
        src.to(device)
        src_mask.to(device)
        logits, loss = model.forward(
            src, src_mask
        )
        #print([vocab.get_itos()[i.detach().cpu().item()] for i,mask in zip(src[0], src_mask[0][400]) if mask])
        prob =F.softmax(logits[0], dim=-1)
        next_char = "".join([vocab.get_itos()[torch.argmax(pchar).item()] for pchar in prob if torch.argmax(pchar).item()!=2 ][len(example)+i+1:])
        if next_char == '</s>':
            break
        generation += next_char
        # print(next_word)
        # ys = torch.cat(
        #     [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        # )
    print(f"shahrukh: {generation}")
    # print([vocab.get_itos()[i.detach().cpu().item()] for i in src[0][:20]])


