import torch
from pathlib import Path
import random
import pandas as pd
import selfies as sf
from multiprocessing import Pool
import re
from collections import OrderedDict, Counter
from torchtext.vocab import vocab as make_vocab
from urllib import request
from multiprocessing import Pool
import os


class DatasetBase:
    def __init__(self, x, vocab, pad_tok="<pad>", mask_tok="<mask>", 
                 mask_freq=0.15, revert_mask_freq=0.1, random_mask_freq=0.1):
        
        # self.batch_size = batch_size
        self.mask_freq = mask_freq
        self.revert_mask_freq = revert_mask_freq
        self.random_mask_freq = random_mask_freq
        self.x = [vocab(i) for i in x]
        self.pad_tok = vocab([pad_tok])[0]
        self.mask_tok = vocab([mask_tok])[0]
        self.toks = torch.LongTensor([v for k, v in vocab.get_stoi().items() if k not in [pad_tok, mask_tok]])
        
    # def __len__(self):
    #     return len(self.x)
    
    # def __getitem__(self, idx):
    #     x = self.x[idx]
    #     return x
    
    def collate_fn(self, batch):
        max_len = max([len(x) for x in batch])
        for x in batch:
            x.extend([self.pad_tok] * (max_len - len(x)))
        x = torch.LongTensor(batch)
        y = torch.clone(x)
        
        padding = x == self.pad_tok
        # generate the token mask
        mask = (torch.rand_like(x.float()) <= self.mask_freq) & (~padding)
        
        # 10% of the "masked" tokens are kept as the original token
        x[(torch.rand_like(x.float()) <= (1 - self.revert_mask_freq)) & mask] = self.mask_tok
        
        # 10% of the masked tokens are replaced with random tokens
        random_mask = (torch.rand_like(x.float()) <= self.random_mask_freq) & mask
        num = random_mask.sum()
        # TODO: maybe change this to sample based on distribution of tokens
        random_toks = self.toks.gather(0, torch.randint(0, len(self.toks), (num,)))
        x[random_mask] = random_toks

        # only need the values for the masked elements
        y = y[mask]
        
        return x, y, mask, padding


class MapDataset(DatasetBase, torch.utils.data.Dataset):
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        return x


class InfiniteDataset(DatasetBase, torch.utils.data.IterableDataset):
    def __iter__(self):
        logged = False
        while True:
            item = random.choice(self.x)
            # if not logged:
            #     print("Chose", item)
            #     logged = True
            yield item


def load_zinc20(datapath=None, train_split=0.9, dev=True):
    # shuffle based on zinc20 tranche.
    # molecules from same tranche are in same train/test split
    files = list(Path(datapath).rglob("*.txt"))
    random.Random(42).shuffle(files)
    split = int(len(files) * train_split)
    train_files = files[:split]
    val_files = files[split:]
    
    if dev:
        # use small subset of the dataset for dev
        train_files = train_files[:(len(train_files) // 20) or 1]
        val_files = val_files[:(len(val_files) // 20) or 1]
        
    def load_files(files):
        rows = []
        cols = None
        for file in files:
            with open(file, "r") as f:
                lines = f.read().splitlines()
            lines = [l.split() for l in lines]
            if cols is None: cols = lines[0]
            rows.extend(lines[1:])
        return pd.DataFrame(rows, columns=cols)
    
    return load_files(train_files), load_files(val_files)


def parse_selfies_file(file):
    with open(file, "r") as f:
        selfies = [list(sf.split_selfies(l.strip())) for l in f.readlines()]
    return selfies


def load_zinc20_smiles(datapath=None, train_split=0.9, dev=True):
    # shuffle based on zinc20 tranche.
    # molecules from same tranche are in same train/test split
    files = list(Path(datapath).rglob("*.txt"))
    random.Random(42).shuffle(files)
    split = int(len(files) * train_split)
    train_files = files[:split]
    val_files = files[split:]
    
    if dev:
        # use 10% of the dataset for dev
        # train_files = train_files[:len(train_files) // 10]
        # val_files = val_files[:len(val_files) // 10]
        train_files = train_files[:10]
        val_files = val_files[:2]
        
    def load_files(files):
        rows = []
        with Pool(processes=min(os.cpu_count() // 2, 32, len(files))) as p:
            for i, r in enumerate(p.imap(parse_selfies_file, files), 1):
                rows.extend(r)
                print(f"\r{i}/{len(files)}", end="")
        print()
        return rows
            
        # return pd.DataFrame(rows, columns=cols)
    
    return load_files(train_files), load_files(val_files)


# def _process_single_smiles(x, raise_exception=False):
#     try:
#         selfies = sf.encoder(x)
#     except sf.EncoderError as e: 
#         if raise_exception:
#             raise e
#         else:
#             return None
#     # Need to come up with better representation
#     return re.split("[\[\]]", selfies)[1::2]

# def smiles_to_selfies(smiles: list, processes=4):
#     if processes <= 1:
#         return [_process_single_smiles(s) for s in smiles]
#     else:
#         processed = []
#         with Pool(processes=processes) as p:
#             for x in p.imap(_process_single_smiles, smiles):
#                 if x:
#                     processed.append(x)
#     return processed


# def convert_files():
#     import glob
#     for file in glob.glob()


def get_vocab(x: list, specials: list, min_freq: int = 10):

    token_freqs = OrderedDict(Counter([i for j in x for i in j]))
    token_freqs = OrderedDict(item for item in token_freqs.items() if item[1] >= min_freq)
    
    vocab = make_vocab(token_freqs, specials=specials, special_first=False, min_freq=10)
    vocab.set_default_index(len(vocab))

    for idx, token in enumerate(token_freqs):
        assert vocab[token] == idx
    
    return vocab, token_freqs


def get_dataloaders(path, pad_tok="<pad>", mask_tok="<mask>", oov_tok="<unk>", dataset_kwargs=None, train_args=None, test_args=None, distributed=False):
    # x_train, x_val = load_zinc20(path)
    print("Loading data")
    x_train, x_val = load_zinc20_smiles(path)
    print(f"{len(x_train):,} Train examples. {len(x_val):,} Val examples.")
    
    # x_train = smiles_to_selfies(x_train['smiles'])
    # x_val = smiles_to_selfies(x_val['smiles'])
    
    print("Generating vocab")
    vocab, token_freqs = get_vocab(x_train, specials=[oov_tok, pad_tok, mask_tok])
    vocab.set_default_index(vocab[oov_tok])
    
    print("Generating dataset")
    train_dataset = InfiniteDataset(x_train, vocab, pad_tok, mask_tok, **(dataset_kwargs or {}))
    val_dataset = MapDataset(x_val, vocab, pad_tok, mask_tok, **(dataset_kwargs or {}))

    train_dataloader_kwargs = train_args or {}
    test_dataloader_kwargs = test_args or {}

    train_defaults = {"num_workers": 8, "batch_size": 16, "prefetch_factor": 2, "persistent_workers": True}
    if isinstance(train_dataset, MapDataset):
        train_defaults["shuffle"] = True

    val_defaults = {"batch_size": 16}
    for k, v in train_defaults.items():
        if k not in train_dataloader_kwargs:
            train_dataloader_kwargs[k] = v
    for k, v in val_defaults.items():
        if k not in test_dataloader_kwargs:
            test_dataloader_kwargs[k] = v
        
    print("Generating dataloaders")
    print("Train:", train_dataloader_kwargs)
    print("Test:", test_dataloader_kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, **train_dataloader_kwargs)
    # If using DDP, create a DistributedSampler to split the Val set across GPUs.
    # Train set is using InfiniteDataset so no need for sampler on the train set
    val_sampler = torch.utils.data.distributed.DistributedSampler(x_val) if distributed else None

    val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=val_dataset.collate_fn, **test_dataloader_kwargs, sampler=val_sampler)

    return train_loader, val_loader, vocab, token_freqs

def _download(args):
    url, dest = args
    if Path(dest).exists():
        return
    # response = request.urlretrieve(url, dest)
    # print(url, dest)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    request.urlretrieve(url, dest)


# def download_zinc20(tranches_file_path, dest_path, processes):
#     # TODO: add POST request to https://zinc20.docking.org/tranches/download
#     tranches_file_path = Path(tranches_file_path, "ZINC-downloader-2D-txt.uri")
#     with open(tranches_file_path, "r") as f:
#         urls = [line.strip() for line in f.readlines()]
    
#     args = [(url, Path(dest_path, *Path(url).parts[-2:])) for url in urls]
#     with Pool(processes=processes) as p:
#         for i, _ in enumerate(p.imap_unordered(_download, args), 1):
#             print(f"\rDownloading {i}/{len(args)}", end="")
#     print()


def process1(file):
    out_file = file.replace("LLcheM/zinc20/", "LLcheM/SELFIES/")
    if os.path.exists(out_file):
        return

    processed = []
    os.makedirs(out_file.rsplit("/", 1)[0], exist_ok=True)

    with open(file, "r") as infile:
        for line in infile.readlines()[1:]:
            try:
                smiles = line.split("\t")[0]
                selfies = sf.encoder(smiles)
                tokens = list(sf.split_selfies(selfies))
                if len(tokens) <= 1024:
                    processed.append(selfies)
            except:
                pass

    with open(out_file, "w") as outfile:
        for line in processed:
            outfile.write(line)
            outfile.write("\n")


def download_zinc20(tranches_file_path, dest_path, processes):
    # TODO: add POST request to https://zinc20.docking.org/tranches/download
    tranches_file_path = Path(tranches_file_path, "ZINC-downloader-2D-txt.uri")
    with open(tranches_file_path, "r") as f:
        urls = [line.strip() for line in f.readlines()]
    
    args = [(url, Path(dest_path, *Path(url).parts[-2:])) for url in urls]
    random.shuffle(args)
    
    for i, arg in enumerate(args, 1):
        print(f"\rDownloading {i}/{len(args)}", end="")
        try:
            _download(arg)
        except KeyboardInterrupt:
            break
        except:
            pass
    
    print()
