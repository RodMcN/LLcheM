import requests
import json
from typing import List, Optional, Union
import random
from pathlib import Path

def get_tranche_ids() -> List[str]:
    """
    Retrieves a list of zinc20 tranche IDs either from a 
    local file tranche_ids.txt or gets all from zinc20 website.
    Tranche IDs are "AAAA", "AAAB", "AAAC", "AAAD", etc.
    """
    tranches_path = Path("tranche_ids.txt")

    if tranches_path.exists():
        with open(tranches_path, "r") as f:
            tranches = [l.strip() for l in f.readlines()]
        return tranches

    else:
        r = requests.get("https://zinc20.docking.org/tranches/all2D.json")
        tranches = json.loads(r.text)
        tranches = [t['name'] for t in tranches]
        with open(tranches_path, "w") as f:
            for t in tranches:
                f.write(t)
                f.write("\n")

        return tranches


def download_tranche(tranche_id: str, dst: Union[Path, str]):
    """
    Downloads a tranche from zinc20. Each tranche is a tsv file containing SMILES and other data.
    Only SMILES data is kept and is saved to <dst>/<trache_id>.txt with 1 SMILES per line.
    """
    dst = Path(dst, f"{tranche_id}.txt")
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)

    url = f"http://files.docking.org/2D/{tranche_id[:2]}/{tranche_id}.txt"
    r = requests.get(url)
    with open(dst, "w") as f:
        for line in r.content.splitlines()[1:]:
            f.write(line.decode().split("\t")[0])
            f.write("\n")


def download_zinc20(output_path: Union[Path, str], max_tranches: Optional[int] = 1000):
    """
    Download multiple zinc20 tranches to output_path. Each tranche may be a different size.
    """
    tranches = get_tranche_ids()
    print(f"{len(tranches):,} tranches")
    if max_tranches:
        tranches = random.Random(42).sample(tranches, k=max_tranches)
    random.shuffle(tranches)
    for i, t in enumerate(tranches, 1):
        download_tranche(t, output_path)
        print(f"\rDownloaded {i}/{len(tranches)} ZINC20 tranches", end="")
    print()