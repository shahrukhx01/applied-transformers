import torch
from transformer_architectures.vanilla.data.batch import subsequent_mask
from torch.nn.functional import pad


def random_inference(test_model):
    #make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    print("Example Untrained Model Prediction:", ys)

def inference_from_pretrained(model, example, vocab, src_pipeline, 
                              device, max_padding = 72, pad_id=2):
    
    model.eval()
    model.to(device)
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    vocab(src_pipeline(example)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        ).unsqueeze(0)
    src = pad(
                src,
                (
                    0,
                    max_padding - len(src),
                ),
                value=pad_id,
            )
    src_mask = torch.ones(1, 1, src.shape[1])
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)
    answer = "shahrukh:"
    for i in range(100):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        next_word_text = vocab.get_itos()[next_word.item()]
        if next_word_text == "</s>":
            break
        elif next_word_text == "<unk>":
            continue
        answer += " " + next_word_text
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    #print("Example Untrained Model Prediction:", ys)
    print(f"text: {example}")
    print(answer)

if __name__ == "__main__":
    for _ in range(10):
        random_inference()
