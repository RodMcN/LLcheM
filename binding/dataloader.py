from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import numpy as np
import torch
from pathlib import Path
import lmdb
import pickle


def load_dataframe(csv_path, target_col, inlcude_ph=False, include_temp=False):
    cols = ["Ligand ID", "Target ID", "split", target_col]
    if inlcude_ph:
        cols.append("pH")
    if include_temp:
        cols.append("Temp (C)")
    df = pd.read_csv(csv_path, sep="\t", dtype={"Ligand ID": str, "Target ID": str})
    df = df[cols].dropna()
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce') # TODO delet
    return df


class Normalise:
    def __init__(self, log_transform=True, epsilon=1e-12):
        self.std = None
        self.mean = None
        self.log_transform = log_transform
        self.epsilon = epsilon
        
    def _check_and_convert(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        return data

    def fit_transform(self, data):
        data = self._check_and_convert(data)
        if self.log_transform:
            data = np.log(data + self.epsilon)
        self.mean = np.mean(data)
        self.std = np.std(data)
        if self.std == 0:
            raise ValueError("Standard deviation is zero, constant data cannot be scaled.")
        data = (data - self.mean) / self.std
        return data

    def transform(self, data):
        if self.std is None or self.mean is None:
            raise Exception("The 'fit_transform' method must be called to compute transformation parameters before using this method.")
        data = self._check_and_convert(data)
        if self.log_transform:
            data = np.log(data + self.epsilon)
        data = (data - self.mean) / self.std
        return data

    def inverse_transform(self, data):
        if self.std is None or self.mean is None:
            raise Exception("The 'fit_transform' method must be called to compute transformation parameters before using this method.")
        data = self._check_and_convert(data)
        data = data * self.std + self.mean
        if self.log_transform:
            data = np.exp(data)
        return data


class BindingDataset(Dataset):
    def __init__(self, df, proteins_dir, ligands, concat=True):
        proteins_dir = Path(proteins_dir)
        ligands = Path(ligands)

        self.df = df
        self.proteins_dir = Path(proteins_dir)
        self.concat = concat

        if ligands.name.endswith(".pt"):
            # ligands is a dict of {ligand_id: torch.tensor}
            self.ligands = torch.load(ligands)
            ids = set(df['Ligand ID'].values)
            self.ligands = {k: v for k, v in self.ligands.items() if k in ids}
            self._is_lmdb = False      
        elif ligands.name.endswith(".lmdb"):
            self.env = lmdb.open(str(ligands), readonly=True, lock=False, readahead=False, meminit=False)
            self._is_lmdb = True
        else:
            raise ValueError

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        protein_id = item['Target ID']
        ligand_id = item['Ligand ID']

        # protein_embedding is a .pt file stored in proteins_dir
        protein_embedding = self.proteins_dir / f"{protein_id}.pt"
        
        if not protein_embedding.exists():
            idx = torch.randint(0, len(self), (1,)).item()
            return self[idx]
        
        else:
            protein_embedding = torch.load(protein_embedding, map_location="cpu")
        
            if self._is_lmdb:
                with self.env.begin(write=False) as txn:
                    ligand_embedding = pickle.loads(txn.get(ligand_id.encode('utf-8')))
            else:
                ligand_embedding = self.ligands[str(item['Ligand ID'])]
            target_val = torch.FloatTensor([item['target']])
    
            if self.concat:
                embedding = torch.cat((protein_embedding, ligand_embedding), -1)
                return embedding, target_val
            else:
                return protein_embedding, ligand_embedding, target_val


def get_dataloaders(csv_path, target_col, proteins_dir, ligands_file, concat, train=True, val=True, test=False, val_split=0.2, train_loader_args=None, val_loader_args=None, test_loader_args=None):
    df = load_dataframe(csv_path, target_col)

    train_data = df[df['split'] == "train"]
    val_data = train_data.sample(int(len(train_data) * val_split), random_state=42)
    train_data = train_data.drop(val_data.index)
    test_data = df[df['split'] == "test"]

    normalise = Normalise()
    train_data['target'] = normalise.fit_transform(train_data[target_col])

    loaders = []
    if train:
        train_dataset = BindingDataset(train_data, proteins_dir, ligands=ligands_file, concat=concat)
        train_loader = DataLoader(train_dataset, **train_loader_args)
        loaders.append(train_loader)
    if val:
        val_data['target'] = normalise.transform(val_data[target_col])
        val_dataset = BindingDataset(val_data, proteins_dir, ligands=ligands_file, concat=concat)
        val_loader = DataLoader(val_dataset, **val_loader_args)
        loaders.append(val_loader)
    if test:
        test_data['target'] = normalise.transform(test_data[target_col])
        test_dataset = BindingDataset(test_data, proteins_dir, ligands=ligands_file, concat=concat)
        test_loader = DataLoader(test_dataset, **test_loader_args)
        loaders.append(test_loader)

    return tuple(loaders), normalise
    
