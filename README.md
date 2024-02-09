# LLcheM (work in progress)
Language model trained on (almost*) 500 million SELFIES chemical representations

\* trained on 484,316,393 molecules

---
## Summary
- Transformer encoder model trained with masked language modelling ([BERT](https://arxiv.org/abs/1810.04805) / [RoBERTa](https://arxiv.org/abs/1907.11692))
- Trained on a subset of the ZINC20 database ([ZINC20 paper](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00675) | [database](https://zinc20.docking.org/))
- SMILES representations from ZINC20 converted to SELFIES ([SELFIES paper](https://iopscience.iop.org/article/10.1088/2632-2153/aba947) | [blog post](https://aspuru.substack.com/p/molecular-graph-representations-and) | [GitHub](https://github.com/aspuru-guzik-group/selfies))

#### LLcheM applications 
- [Will-It-Bind](https://github.com/RodMcN/LLcheM#will-it-bind) predicts protein-ligand binding affinities from LLcheM and [ESM2](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf)
---

## Models
- Transformer encoder models, similar to BERT, generate embeddings for small molecules.
- Input is a SELFIES or SMILES string
  - SMILES are converted to SELFIES prior to passing to the model

### Training notes
- Trained using AdamW optimiser
- Linear LR warmup with cosine LR decay
- Loss function is weighted by inverse frequency of atoms on the training set - predictions for more common atoms (such as carbon) contributes less to loss
- Models are trained for up to 1 million batch updates with early stopping.
- Each batch contains 1024 SELFIES strings, containing on average approx. 20k tokens, ~15% of which are masked training tokens.
  - Batches are optionally split agross multiple GPUs and gradient accumulation steps. 
- Models are trained on xyz SELFIES strings and tested on acb SELFIES strings.
- ZINC20 data is split by tranche before splitting into train and validation data.

### Models:

|                            | S0-S | S1-S | M0-S | M1-S | M1-E | M1-R | L0-S | L0-R | L1-S |
| ---------------------------| --   | ---- | ---- | ---- | -    | -    | ---  | ---- | -    |
| Number of params (approx.) | 215K | -    | 4.8M | 32M  | 32M  | 32M  | 200M | 200M | 250M |
| Embedding Dim              | 64   | 128  | 256  | 512  | 512  | 512  | 1024 | 1024 | 1024 |
| Number of layers           | 4    | 4    | 6    | 10   | 10   | 10   | 16   | 16   | 20   |
| Attention heads            | 4    | 4    | 8    | 8    | 8    | 8    | 16   | 16   | 16   |
| Pos encoding*              | S    | S    | S    | S    | E    | R    | S    | R    | S    |
| Validation perplexity      | 1.590| -    | 1.247| 1.189| 1.185| 1.182| -    | -    | -    |

\* S = [Sinusoidal encoding](https://arxiv.org/abs/1706.03762),
R = [Rotary Positional Embedding (RoPE)](https://arxiv.org/abs/2104.09864), E = learned positional embeddings.

The token distribution in the dataset is highly skewed and the perplexity calculation does not take this skewness into account.


### plot of loss curves
### plot of embeddings

---

# Will it Bind?
LLcheM application: Predicting protein-ligand binding affinities
---
Using LLcheM and ESM2, predict binding affinities of proteins and ligands.
- LLcheM generates molecule embeddings.
- ESM2 ([paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf) | [code](https://github.com/facebookresearch/esm)) generates protein embeddings.
- Trained on data from [BindingDB](https://www.bindingdb.org).
- 2 model architectures - Will-it-Bind Lite, a fully connected MLP and Will-it-Bind X, a transformer
### Will-it-Bind Lite
Embedded sequences (amino acids and SELFIES) are averaged across sequence dim and concatenated, a fully connected MLP is used to predict affinities

![Will it bind - Lite](https://raw.githubusercontent.com/RodMcN/media/main/LLcheM/willitbind_lite.png)

### Will-it-Bind X
A transformer using cross attention on the embedded sequences to predict binding affinity

![Will it bind - X](https://raw.githubusercontent.com/RodMcN/media/main/LLcheM/willitbind_x.png)

### Training
- Models trained to predict IC50
  - Training data is log-transformed and standardised to 0 mean and unit standard deviation. Outputs can be inverse transformed to generate IC50 prediction.
- Hyperparameters for both models were optimised with [Optuna]("https://optuna.readthedocs.io/en/stable/index.html")
- Trained on xyz protein-ligand pairs from BindingDB

| model | num params | MAE | R^2 score |
|-------|------------|-----|-------------|
| Lite  | 50         | -   | -           |
| X     | 100        | -   | -           |
- MAE and R^2 calculated after inverse-transforming model outputs from log standardised to actual IC50.

Limitations:
- Will-It-Bind was trained on the `BindingDB Target Chain Sequence` from BindingDB which, in most cases, does not include the entire target protein.
- `<trunk>` tokens are not appended to truncated sequences prior to ESM embedding.