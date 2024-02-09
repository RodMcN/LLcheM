import selfies as sf
from multiprocessing import Pool
from pathlib import Path
from tqdm.auto import tqdm
import os

def convert_file_to_selfies(args_tuple):
    in_file, out_file, max_len, cleanup = args_tuple

    out_file = Path(out_file)
    if out_file.exists():
        return 0
    
    num_selfies = 0
    with open(in_file, "r") as infile, open(out_file, "w") as out_file:
        while (line := infile.readline().strip()):
            try:
                smiles = line.split("\t")[0]
                selfies = sf.encoder(smiles)
                if max_len and len(list(sf.split_selfies(selfies))) > max_len:
                    continue
                else:
                    out_file.write(selfies)
                    out_file.write("\n")
                    num_selfies += 1
                    if num_selfies > 5_000_000:
                        break
            except:
                pass
    if cleanup:
        os.remove(in_file)
    return num_selfies


def convert_multiple_files(in_dir, out_dir, max_len=1000, cleanup=False, n_processes=16):
    in_dir = Path(in_dir)
    
    args = []
    for file in in_dir.rglob("*.txt"):
        out_file = str(file).replace(str(in_dir), str(out_dir))
        args.append((file, out_file, max_len, cleanup))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    import random
    random.shuffle(args)
    
    num_selfies = 0
    with Pool(processes=n_processes) as p:
        tqdm_iterator = tqdm(p.imap_unordered(convert_file_to_selfies, args), total=len(args))
        for i, n in enumerate(tqdm_iterator, 1):
            num_selfies += n
            tqdm_iterator.set_description(f"Converted {i} files ({num_selfies:,} selfies)")
