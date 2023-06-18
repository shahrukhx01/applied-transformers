"""
---
title: BERT Embeddings of chunks of text
summary: >
  Generate BERT embeddings for chunks using a frozen BERT model
---

# BERT Embeddings of chunks of text

This is the code to get BERT embeddings of chunks for [RETRO model](index.html).
"""

from typing import List
import torch
from transformers import BertTokenizer, BertModel


class BERTChunkEmbeddings:
    """
    ## BERT Embeddings

    For a given chunk of text $N$ this class generates BERT embeddings $\text{B\small{ERT}}(N)$.
    $\text{B\small{ERT}}(N)$ is the average of BERT embeddings of all the tokens in $N$.
    """

    def __init__(self, device: torch.device):
        self.device = device

        # Load the BERT tokenizer from [HuggingFace](https://huggingface.co/bert-base-uncased)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                                                          
        # Load the BERT model from [HuggingFace](https://huggingface.co/bert-base-uncased)
        self.model = BertModel.from_pretrained("bert-base-uncased")

        # Move the model to `device`
        self.model.to(device)

    @staticmethod
    def _trim_chunk(chunk: str):
        """
        In this implementation, we do not make chunks with a fixed number of tokens.
        One of the reasons is that this implementation uses character-level tokens and BERT
        uses its sub-word tokenizer.

        So this method will truncate the text to make sure there are no partial tokens.

        For instance, a chunk could be like `s a popular programming la`, with partial
        words (partial sub-word tokens) on the ends.
        We strip them off to get better BERT embeddings.
        As mentioned earlier this is not necessary if we broke chunks after tokenizing.
        """
        # Strip whitespace
        stripped = chunk.strip()
        # Break words
        parts = stripped.split()
        # Remove first and last pieces
        stripped = stripped[len(parts[0]):-len(parts[-1])]

        # Remove whitespace
        stripped = stripped.strip()

        # If empty return original string
        if not stripped:
            return chunk.strip()
        # Otherwise, return the stripped string
        else:
            return stripped

    def __call__(self, chunks: List[str]):
        """
        ### Get $\text{B\small{ERT}}(N)$ for a list of chunks.
        """

        # We don't need to compute gradients
        with torch.no_grad():
            # Trim the chunks
            trimmed_chunks = [self._trim_chunk(c) for c in chunks]

            # Tokenize the chunks with BERT tokenizer
            tokens = self.tokenizer(trimmed_chunks, return_tensors='pt', add_special_tokens=False, padding=True)

            # Move token ids, attention mask and token types to the device
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            token_type_ids = tokens['token_type_ids'].to(self.device)
            # Evaluate the model
            output = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

            # Get the token embeddings
            state = output['last_hidden_state']
            # Calculate the average token embeddings.
            # Note that the attention mask is `0` if the token is empty padded.
            # We get empty tokens because the chunks are of different lengths.
            emb = (state * attention_mask[:, :, None]).sum(dim=1) / attention_mask[:, :, None].sum(dim=1)

            return emb


def _test():
    """
    ### Code to test BERT embeddings
    """

    # Initialize
    device = torch.device('cpu')
    bert = BERTChunkEmbeddings(device)

    # Sample
    text = ["Replace me by any text you'd like.",
            "Second sentence"]

    # Check BERT tokenizer
    encoded_input = bert.tokenizer(text, return_tensors='pt', add_special_tokens=False, padding=True)


    # Check BERT model outputs
    output = bert.model(input_ids=encoded_input['input_ids'].to(device),
                        attention_mask=encoded_input['attention_mask'].to(device),
                        token_type_ids=encoded_input['token_type_ids'].to(device))
    state = output['last_hidden_state']
    attention_mask = encoded_input['attention_mask']
    emb = (state * attention_mask[:, :, None]).sum(dim=1) / attention_mask[:, :, None].sum(dim=1)

if __name__ == '__main__':
    _test()