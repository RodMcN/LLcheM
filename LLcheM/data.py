import torch
from pathlib import Path
import random
# import pandas as pd
import selfies as sf
from multiprocessing import Pool
import re
from collections import OrderedDict, Counter
from torchtext.vocab import vocab as make_vocab
from urllib import request
from multiprocessing import Pool
import os
import gc
import lmdb
from typing import Optional
from tqdm.auto import tqdm
import pickle


class BaseDataset:
    def __init__(self, x, vocab, lmdb_path=None, pad_tok="<pad>", mask_tok="<mask>", 
                 mask_freq=0.15, revert_mask_freq=0.1, random_mask_freq=0.1):
        """
        Base class for datasets. Use MapDataset or InfiniteDataset instead.
        if lmdb_path is None, x is list of parsed selfies, each parsed selfies is a list
        if lmbd_path is a str/Path, x is a list of indces.
        :param mask_freq: proportion of token that are masked
        :param revert_mask_freq: proportion of the masked tokens that are kept as the original token
        :param random_mask_freq: proportion of he masked tokens that are replaced with random tokens
        """
        # self.batch_size = batch_size
        self.mask_freq = mask_freq
        self.revert_mask_freq = revert_mask_freq
        self.random_mask_freq = random_mask_freq

        self.is_lmdb = bool(lmdb_path)
        self.txn = None
        
        if not self.is_lmdb:
            self.x = [vocab(i) for i in x]
        
        else:
            self.x = x
            self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
            self.vocab = vocab
        
        self.pad_tok = vocab([pad_tok])[0]
        self.mask_tok = vocab([mask_tok])[0]
        self.toks = torch.LongTensor([v for k, v in vocab.get_stoi().items() if k not in [pad_tok, mask_tok]])

    
    def get_item(self, idx):
        if not self.is_lmdb:
            return self.x[idx]
        else:
            # key = self.x[idx]
            key = idx
            key = str(key).encode('ascii')
            value = self.txn.get(key)#.decode("ascii")
            value = pickle.loads(value)
            # value = list(sf.split_selfies(value))
            value = self.vocab(value)
            return value
    

    def init_worker(self):
        if self.is_lmdb:
            self.txn = self.env.begin(write=False)
    
    def close(self):
        if self.txn is not None:
            self.txn.abort()
            self.env.close()
    
    def __del__(self):
        self.close()


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


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.init_worker()


class MapDataset(BaseDataset, torch.utils.data.Dataset):
    """
    A map-style dataset wrapper for BaseDataset.
    Implements __getitem__() and __len__() and represents a map from indices to data samples.
    """
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.get_item(idx)


class InfiniteDataset(BaseDataset, torch.utils.data.IterableDataset):
    """
    An iterable-style dataset wrapper for BaseDataset for infinite iteration over the data.
    During each iteration, it provides a random sample from the dataset. Useful when training on tens of millions of 
    zinc20 samples to avoid epoch based training regimes.
    Each call yields a random sample from the dataset.
    """    
    def __iter__(self):
        if self.is_lmdb:
            while True:
                # choose a random index then return lmdb value for the corresponding index
                idx = random.choice(self.x)
                yield self.get_item(idx)
        else:
            while True:
                # all items are stored in memory
                # just return an item from self.x
                item = random.choice(self.x)
                yield item


def load_zinc20(datapath=None, train_split=0.9, dev=False):
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


def load_zinc20_selfies(datapath=None, train_split=0.9, dev=False, max_len=1000):
    # shuffle based on zinc20 tranche.
    # molecules from same tranche are in same train/test split
    files = list(Path(datapath).rglob("*.txt"))
    random.Random(42).shuffle(files)
    split = int(len(files) * train_split)
    train_files = files[:split]
    val_files = files[split:]
    
    if dev:
        # use 10% of the dataset for dev
        # train_files = train_files[:len(train_files) // 20]
        # val_files = val_files[:len(val_files) // 20]
        train_files = train_files[:10]
        val_files = val_files[:2]
        
    def load_files(files):
        rows = []
        with Pool(processes=min(os.cpu_count() // 2, 32, len(files))) as p:
            for i, r in enumerate(p.imap_unordered(parse_selfies_file, files), 1):
                if max_len:
                    for x in r:
                        if len(x) <= max_len:
                            rows.append(x)
                else:
                    rows.extend(r)
                print(f"\rloading SELFIES files:{i}/{len(files)} ({len(rows):,} rows)", end="")
                if i % 10 == 0:
                    gc.collect()
        print()
        return rows
            
        # return pd.DataFrame(rows, columns=cols)
    
    return load_files(train_files), load_files(val_files)


def get_vocab(lmdb_path: Optional[str], x: Optional[list], specials: list, min_freq: int = 200):
    """
    if lmdb_path is None, x is a list if lists of SELFIES tokens.
    if lmdb_path is not None, x should be a list of training indices
    """
    if lmdb_path:
        token_freqs = Counter()
        x = set(x)
        env = lmdb.open(lmdb_path, readonly=True)
        try:
            with env.begin() as txn:
                numel = int(txn.get("__len__".encode("ascii")).decode("ascii"))

                cursor = txn.cursor()
                for (key, value) in tqdm(cursor, total=numel, desc="Generating vocab"):
                    key = key.decode("ascii")
                    if (not key.startswith("_")) and (int(key) in x):
                        # only count tokens from training set
                        # token_freqs.update(sf.split_selfies(value.decode("ascii")))
                        token_freqs.update(pickle.loads(value))
        finally:
            env.close()

    else:
        token_freqs = OrderedDict(Counter([i for j in x for i in j]))
    
    token_freqs = OrderedDict(item for item in token_freqs.items() if item[1] >= min_freq)
    
    vocab = make_vocab(token_freqs, specials=specials, special_first=False, min_freq=min_freq)

    for idx, token in enumerate(token_freqs):
        assert vocab[token] == idx
    
    return vocab, token_freqs


def get_dataloaders(path, train_split=0.9, pad_tok="<pad>", mask_tok="<mask>", oov_tok="<unk>", dataset_kwargs=None, train_args=None, test_args=None, distributed=False, use_lmdb=False):
    print("Loading data")
    if not use_lmdb:
        # path is a path to folder containing SELFIES txt files to be loaded into memory
        # x_train and x_val are parsed SELFIES stored in memory
        x_train, x_val = load_zinc20_selfies(path, train_split=train_split)
        print(f"{len(x_train):,} Train examples. {len(x_val):,} Val examples.")
        print(f"{sum([len(x) for x in x_train]):,} Train tokens")
        
        print("Generating vocab")
        vocab, token_freqs = get_vocab(lmdb_path=None, x=x_train, specials=[oov_tok, pad_tok, mask_tok])
    else:
        # path is a path to a lmdb
        env = lmdb.open(path, readonly=True, lock=False)
        try:
            with env.begin() as txn:
                numel = int(txn.get("__len__".encode("ascii")).decode("ascii"))
                print(f"{numel=}")
        finally:
            env.close()

        indices = list(range(numel))
        split = int(numel * train_split)
        # x_train and x_val are lists of keys in lmdb
        x_train = indices[:split]
        x_val = indices[split:]
        print(f"{len(x_train):,} Train examples. {len(x_val):,} Val examples.")
        vocab, token_freqs = get_vocab(lmdb_path=path, x=x_train, specials=[oov_tok, pad_tok, mask_tok])

    vocab.set_default_index(vocab[oov_tok])

    print("Generating dataset")
    lmdb_path = path if use_lmdb else None
    train_dataset = InfiniteDataset(x_train, vocab, lmdb_path=lmdb_path, pad_tok=pad_tok, mask_tok=mask_tok, **(dataset_kwargs or {}))
    val_dataset = MapDataset(x_val, vocab, lmdb_path=lmdb_path, pad_tok=pad_tok, mask_tok=mask_tok, **(dataset_kwargs or {}))

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
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, worker_init_fn=worker_init_fn, **train_dataloader_kwargs)
    # If using DDP, create a DistributedSampler to split the Val set across GPUs.
    # Train set is using InfiniteDataset so no need for sampler on the train set
    val_sampler = torch.utils.data.distributed.DistributedSampler(x_val) if distributed else None

    val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=val_dataset.collate_fn, worker_init_fn=worker_init_fn, **test_dataloader_kwargs, sampler=val_sampler)

    return train_loader, val_loader, vocab, token_freqs
