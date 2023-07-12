# LLcheM (work in progress)
Language model trained on SELFIES chemical representations
---
## Summary
- Encoder transformer trained with masked language modelling ([BERT](https://arxiv.org/abs/1810.04805) / [RoBERTa](https://arxiv.org/abs/1907.11692))
- Trained on a subset of ZINC20 database ([ZINC20 paper](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00675) | [database](https://zinc20.docking.org/))
- SMILES representations from ZINC20 converted to SELFIES ([SELFIES paper](https://iopscience.iop.org/article/10.1088/2632-2153/aba947) | [blog post](https://aspuru.substack.com/p/molecular-graph-representations-and) | [GitHub](https://github.com/aspuru-guzik-group/selfies))

---

### Test Run
- ~38M params
- 512D embedding, 12 encoder layers, 8 attention heads per layer
- Sinusoidal positional encoding
- ~15M training molecules (~600M tokens)
- 50K steps with effective batch size 2048 molecules
- Trained using AdamW optimiser
- Linear LR warmup to 0.01 over 10K steps with cosine LR decay
- Loss function is weighted by inverse frequency of atoms on the training set - correctly predicting more common atoms (such as carbon) contributes less to loss
- ZINC20 data is split by tranche before splitting into train and validation data so no molecules from the same trache are in different splits.

![Loss graph](https://raw.githubusercontent.com/RodMcN/media/main/LLcheM/test_run.png)

- perplexity is calculated using pseudo-perplexity method outlined in [ESM2](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf) rather than a true perplexity, which is not well defined for masked language models.

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
