# Applied Transformers (PyTorch)

A playground-like experimental project to explore various transformer architectures from scratch.

## Resources:

### Intuitions:

1. Intuition behind Attention Mechanism | [Notebook](<https://github.com/shahrukhx01/applied-transformers/blob/main/intuitions/0.%20Transformers%20%3E%20Understanding%20Self-Attention%20and%20Cross-Attention-Copy1.ipynb>)
2. Intuition behind individual Transformer Blocks | [Notebook](<https://github.com/shahrukhx01/applied-transformers/blob/main/intuitions/1.%20Transformers%20%3E%20Transformer%20from%20scratch%20(Annotated%20Transformer).ipynb>)
3. Intuition behind Chunked Cross-attention by RETRO Deepmind | [Notebook](<https://github.com/shahrukhx01/applied-transformers/blob/main/intuitions/2.%20Transformers%20%3E%20(Deepmind%20RETRO)%20Chunked%20Cross-Attention.ipynb>)

### Implementations from Scratch:

Create virtual environment:

```bash
conda create -n applied-transformers python=3.10
conda activate applied-transformers
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

1. Transformer Model from Scratch {Vaswani et. al, 2017} | [Dataset Sample](https://github.com/shahrukhx01/applied-transformers/blob/main/transformer_architectures/vanilla/data/sample_data.csv) | [Python Code](https://github.com/shahrukhx01/applied-transformers/tree/main/transformer_architectures/vanilla)

```bash
# example training run
python transformer_architectures/vanilla/run.py --num_layers=5\
 --d_model=256 --d_ff=1024 --num_heads=4 --dropout=0.2 \
--train_path=<PATH_TO_TRAIN_DATASET>.csv  --valid_path=<PATH_TO_VALIDATION_DATASET>.csv
```

2. GPT Model from Scratch {Radford et. al, 2018} | [Coming Soon]()
3. BERT Model from Scratch {Lewis et. al, 2019} | [Coming Soon]()
4. RETRO Model from Scratch {Borgeaud et. al, 2021} | [Coming Soon]()
5. BART Model from Scratch {Lewis et. al, 2019} | [Coming Soon]()

## TODO:

- Text Generation Schemes
- Text Generation Eval Metrics
- Sequence Tokenization Algorithms
- Optimized Einsum Implementation

## References

1. [http://nlp.seas.harvard.edu/annotated-transformer/](http://nlp.seas.harvard.edu/annotated-transformer/)
2. [https://nn.labml.ai/transformers/models.html](https://nn.labml.ai/transformers/models.html)
3. [Transformers from scratch | CodeEmporium](https://www.youtube.com/playlist?list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4)
