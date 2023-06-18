from typing import List, Dict, Optional, Callable
from pathlib import PurePath, Path

import torch


class TextDataset:
    itos: List[str]
    stoi: Dict[str, int]
    n_tokens: int
    train: str
    valid: str
    standard_tokens: List[str] = []

    @staticmethod
    def load(path: PurePath):
        with open(str(path), 'r') as f:
            return f.read()

    def __init__(self, path: PurePath, tokenizer: Callable, train: str, valid: str, test: str, *,
                 n_tokens: Optional[int] = None,
                 stoi: Optional[Dict[str, int]] = None,
                 itos: Optional[List[str]] = None):
        self.test = test
        self.valid = valid
        self.train = train
        self.tokenizer = tokenizer
        self.path = path

        if n_tokens or stoi or itos:
            assert stoi and itos and n_tokens
            self.n_tokens = n_tokens
            self.stoi = stoi
            self.itos = itos
        else:
            self.n_tokens = len(self.standard_tokens)
            self.stoi = {t: i for i, t in enumerate(self.standard_tokens)}

            tokens = self.tokenizer(self.train) + self.tokenizer(self.valid)
            tokens = sorted(list(set(tokens)))

            for t in tokens:
                self.stoi[t] = self.n_tokens
                self.n_tokens += 1

            self.itos = [''] * self.n_tokens
            for t, n in self.stoi.items():
                self.itos[n] = t

    def text_to_i(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text)
        return torch.tensor([self.stoi[s] for s in tokens if s in self.stoi], dtype=torch.long)

    def __repr__(self):
        return f'{len(self.train) / 1_000_000 :,.2f}M, {len(self.valid) / 1_000_000 :,.2f}M - {str(self.path)}'


class TextFileDataset(TextDataset):
    standard_tokens = []

    def __init__(self, path: PurePath, tokenizer: Callable, *,
                 url: Optional[str] = None,
                 filter_subset: Optional[int] = None):
        path = Path(path)
        if not path.exists():
            if not url:
                raise FileNotFoundError(str(path))
            else:
                raise NotImplementedError

        text = self.load(path)
        if filter_subset:
            text = text[:filter_subset]
        split = int(len(text) * .9)
        train = text[:split]
        valid = text[split:]

        super().__init__(path, tokenizer, train, valid, '')