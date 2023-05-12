import pandas as pd
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(file_path, src_column, tgt_column):
    dataset = pd.read_csv(file_path)
    corpus = list(dataset[src_column].values) + list(dataset[tgt_column].values)
    for line in corpus:
        yield word_level_tokenizer(line)

def yield_tokens_char_level(file_path, src_column, tgt_column):
    dataset = pd.read_csv(file_path)
    for _, row in dataset.iterrows():
       yield char_level_tokenizer(row[src_column], row[tgt_column])

def word_level_tokenizer(seq):
    return str(seq).strip().split()

def char_level_tokenizer(src, tgt):
    return list(f"{src} | {tgt}".lower().strip())

yield_vocab_tokenizer_map = {
    "word": yield_tokens,
    "char": yield_tokens_char_level,
}

tokenizer_fn_map = {
    "word": word_level_tokenizer,
    "char": char_level_tokenizer,
}

def build_vocab(file_path, tokenizer_type="word", src_column="source", tgt_column="target"):
    """
    Builds the vocab when the source and target langauge are same.
    You'd need to create separate vocabularies if you are doing NMT,
    or tasks which have non-overlapping vocabs.

    Args:
        file_path (_type_): _description_
        src_column (str, optional): _description_. Defaults to "source".
        tgt_column (str, optional): _description_. Defaults to "target".
    """
    print("Building Vocabulary ...")
    vocab = build_vocab_from_iterator(yield_vocab_tokenizer_map[tokenizer_type](file_path, src_column, tgt_column), 
                                min_freq=2,specials=["<s>", "</s>", "<blank>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab
