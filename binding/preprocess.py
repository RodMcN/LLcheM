import csv
import re
import selfies as sf
import torch
from esm import FastaBatchedDataset, pretrained
from LLcheM.model import LLcheM
from tempfile import TemporaryFile, TemporaryDirectory
import random
from tqdm.auto import tqdm
from pathlib import Path
import requests
from io import BytesIO
from zipfile import ZipFile
import lmdb
import pickle
import gc

def process_bindingdb(bindingdb_path, out_path):
    columns = ["Ligand ID", "Target ID", "Target Name", "Target Source Organism", "Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)", "kon (M-1-s-1)", "koff (s-1)", "pH", "Temp (C)", "Ligand SELFIES", "Target FASTA", "split"]

    pattern = re.compile(R"(\d+(\.\d+)?)")
    rng = random.Random(42)
    
    print("Processing BindingDB")
    with open(bindingdb_path, "r") as in_file, open(out_path, "w") as out_csv:
        writer = csv.writer(out_csv, delimiter="\t")
        
        in_file.readline() # skip forst line
        writer.writerow(columns)

        # extract data from bindingDB
        for line in tqdm(in_file.readlines(), desc="Processing BindingDB"):
            line = line.split("\t")
            ligand_id = line[4]
            target_fasta = line[38]
            # generate a numeric ID for the protein
            target_id = str(abs(hash((target_fasta))))
            target_name = line[6]
            target_organism = line[7]

            # extract Temp, pH, Ki, IC50, etc
            measurements = []
            for m in line[8:16]:
                match = re.match(pattern, m.strip())
                if match:
                    measurements.append(match.group(0))
                else:
                    measurements.append("")
            if not any(measurements[:-2]):
                continue

            smiles = line[1]
            try:
                selfies = sf.encoder(smiles)
            except Exception as e:
                continue
            if len(selfies) > 1000:
                continue

            # for simplicity, randomly split into train and test here
            if rng.random() > 0.8:
                split = "test"
            elif rng.random() > 0.5:
                split = "val"
            else:
                split = "train"

                
            row = [ligand_id, target_id, target_name, target_organism, *measurements, selfies, target_fasta, split]
            writer.writerow(row)


def generate_llchem_tok_embeddings(model_path, db_path, output_dir, device="cuda"):
    output_dir = Path(output_dir)

    print("Processing BindingDB with LLcheM")
    model = LLcheM.load_model(model_path, map_location="cpu")
    model.eval()
    model = model.to(device)

    mean = 0
    var = 0
    n = 0

    lmdb_env = lmdb.open(str(output_dir / "train.lmdb"), map_size=250 * (1024**3))
    txn = lmdb_env.begin(write=True)

    ligand_ids = []
    with open(db_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            ligand_ids.append(row['Ligand ID'])
    ligand_ids = set(ligand_ids)


    with open(db_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, row in tqdm(enumerate(reader, 1), desc="Processing SELFIES with LLcheM"):
            ligand_id = row['Ligand ID']
            if ligand_id not in ligand_ids:
                continue
            selfies = row['Ligand SELFIES']
            emb = model.get_embeddings(selfies=selfies, mean_reduce=False).cpu()
            txn.put(ligand_id.encode('utf-8'), pickle.dumps(emb))
            ligand_ids.remove(ligand_id)

            assert not emb.requires_grad

            if row['split'] == "train":
                # iteratively calculate mean and variance of training data
                n += 1
                m = emb.mean(0)
                delta = m - mean
                mean = (delta / n) + mean
                delta2 = m - mean
                var = (delta * delta2) + var
            del emb
            
            if i % 10_000 == 0:
                if "cuda" in device:
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
                txn.commit()
                txn = lmdb_env.begin(write=True)
                gc.collect()
        
        txn.put("mean".encode('utf-8'), pickle.dumps(mean))
        txn.put("std".encode('utf-8'), pickle.dumps(torch.sqrt(var / (n-1))))
        txn.commit()

    del model
    if "cuda" in device:
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


def generate_llchem_embeddings(model_path, db_path, output_dir, device="cuda"):
    out_file = Path(output_dir, "bindingdb_embeddings.pt")
    if out_file.exists():
        print(out_file, "exists")
        return
    print("Processing BindingDB with LLcheM")
    model = LLcheM.load_model(model_path, map_location="cpu")
    model.eval()
    model = model.to(device)

    # there are >1M molecules, so get keys first
    keys = set()
    with open(db_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            keys.add(row['Ligand ID'])


    size = model.encoder.layers[-1].fc2.out_features
    mean = torch.zeros(size)
    var = torch.zeros(size)
    n = 0

    keys.add("mean")
    keys.add("std")
    embeddings = dict.fromkeys(keys)
    with open(db_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in tqdm(reader, desc="Processing SELFIES with LLcheM"):
            ligand_id = row['Ligand ID']
            if embeddings[ligand_id] is not None:
                continue
            selfies = row['Ligand SELFIES']
            # if len(selfies) > 1000:
            #     continue
            emb = model.get_embeddings(selfies=selfies).cpu()
            embeddings[ligand_id] = emb

            # iteratively calculate mean and variance of training data
            if row['split'] == "train":
                n += 1
                delta = emb - mean
                mean += delta / n
                delta2 = emb - mean
                var += delta * delta2

    std = torch.sqrt(var)
    for k, v in embeddings.items():
        if v is not None:
            embeddings[k] = (v - mean) / std

    embeddings['mean'] = mean
    embeddings['std'] = torch.sqrt(var)

    print("Saving to", out_file)
    out_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(embeddings, out_file)
    del model
    if "cuda" in device:
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


def generate_esm_embeddings(model_path, db_path, output_dir, device="cuda"):
    # adapted from https://github.com/facebookresearch/esm/blob/main/scripts/extract.py
    # temp_file = TemporaryFile()
    temp_file = Path("temp_fasta")
    # assert not temp_file.exists()
    seen = set()
    with open(db_path, "r") as f_in, open(temp_file, "w") as f_out:
        f_in.readline()
        for line in f_in.readlines():
            line = line.split("\t")
            protein_id = line[1]
            if protein_id in seen:
                continue
            seen.add(protein_id)
            fasta = line[13]
            if any(f.islower() or f.isnumeric() for f in fasta):
                continue
            f_out.write(f">{protein_id}\n{fasta}\n")


    model, alphabet = pretrained.load_model_and_alphabet(model_path)
    model.eval()
    model = model.to(device)

    toks_per_batch = 512#8192
    truncation_seq_length = 1022

    dataset = FastaBatchedDataset.from_file(temp_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches,
        num_workers=1, pin_memory="cuda" in device
    )

    repr_layer = int(re.findall("esm2_t([0-9]{2})_", model_path)[0])

    with torch.no_grad():
        for labels, strs, toks in tqdm(data_loader):

            toks = toks.to(device=device, non_blocking=True)

            out = model(toks, repr_layers=[repr_layer], return_contacts=False)

            representations = out["representations"][repr_layer]

            output_dir = Path(output_dir)
            for i, label in enumerate(labels):
                output_file1 = output_dir / "per_tok" / f"{label}.pt"
                output_file2 = output_dir / "mean_repr" / f"{label}.pt"
                output_file1.parent.mkdir(parents=True, exist_ok=True)
                output_file2.parent.mkdir(parents=True, exist_ok=True)

                truncate_len = min(truncation_seq_length, len(strs[i]))

                result = representations[i, 1 : truncate_len + 1].clone()
                torch.save(result.cpu().clone(), output_file1)
                torch.save(result.mean(0).cpu().clone(), output_file2)
    
    # temp_file.close()
    del model
    if "cuda" in device:
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()



def download_bindingdb(dest):
    dest = Path(dest)
    url = "https://www.bindingdb.org/bind/downloads/BindingDB_All_202309.tsv.zip"
    
    out = BytesIO()
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_bytes = int(r.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_bytes, unit='iB', unit_scale=True, desc="Downloading BindingDB")
        for data in r.iter_content(1024):
            progress_bar.update(len(data))
            out.write(data)
        progress_bar.close()
    ZipFile(out).extractall(dest)
    
    return dest / "BindingDB_All.tsv"


def main():
    # argparse and such
    # download and extract
    binding_db_path = "/DATA/binding_data.tsv"
    if not Path(binding_db_path).exists():
        with TemporaryDirectory() as tmp_dir:
            tmp_db_path = download_bindingdb(tmp_dir)
            process_bindingdb(tmp_db_path, binding_db_path)

    # Process with LLcheM
    generate_llchem_tok_embeddings("/DATA/LLcheM_out/test_run_1.pt", binding_db_path, "/DATA/binding/ligands")
    # generate_llchem_embeddings("/home/roddy/projects/LLcheM/LLcheM_out/test_run_1.pt", binding_db_path, "/home/roddy/projects/LLcheM/binding/ligands")

    # process with ESM
    # generate_esm_embeddings("esm2_t33_650M_UR50D", binding_db_path, "/DATA/binding/targets/esm2_t33_650M_UR50D")
    # generate_esm_embeddings("esm2_t33_650M_UR50D", binding_db_path, "/home/roddy/projects/LLcheM/binding/targets/esm2_t33_650M_UR50D")

if __name__ == "__main__":
    main()
