import os
import torch
import pandas as pd

from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
from Bio.Data.PDBData import protein_letters_3to1


DATA_ROOT=r"D:\Python311\Pets\GraphT\data"

DIST_THRESHOLD=8.0
MAX_RESIDUES=600
EDGE_CAP=20000

config=ProteinGraphConfig(granularity="CA")

def FASTA_seqgen(sorted_nodes,pdb_id,fasta_dir):

    sequence="".join(
        protein_letters_3to1.get(
            n[1]["residue_name"].upper(), "X"
        )
        for n in sorted_nodes
    )

    fasta_path=os.path.join(fasta_dir, f"{pdb_id}.fasta")

    with open(fasta_path, "w") as f:
        f.write(f">{pdb_id}\n")
        f.write(sequence)


def contact_mapgen(coords, pdb_id, tensor_dir):

    dist=torch.cdist(coords, coords)

    dense_map=dist.to(torch.float16)
    dense_map.fill_diagonal_(0)

    edge_index=(dist < DIST_THRESHOLD).nonzero(as_tuple=False).t()
    mask=edge_index[0] != edge_index[1]
    edge_index=edge_index[:, mask].long()

    if edge_index.shape[1] > EDGE_CAP:
        return False

    tmp_path=os.path.join(tensor_dir, f"{pdb_id}.tmp")
    final_path=os.path.join(tensor_dir, f"{pdb_id}.pt")

    torch.save({
        "V_dense_map": dense_map,
        "G_edge_index": edge_index,
        "G_num_nodes": coords.shape[0]
    }, tmp_path)

    os.replace(tmp_path, final_path)

    return True


def build_protein(pdb_id, fasta_dir, tensor_dir):

    final_path=os.path.join(tensor_dir, f"{pdb_id}.pt")
    if os.path.exists(final_path):
        return

    try:
        graph=construct_graph(
            config=config,
            pdb_code=pdb_id
        )
    except Exception:
        return

    sorted_nodes=sorted(
        graph.nodes(data=True),
        key=lambda x: (x[1]["chain_id"], x[1]["residue_number"])
    )

    coords=torch.tensor(
        [n[1]["coords"] for n in sorted_nodes if "coords" in n[1]],
        dtype=torch.float32
    )

    if coords.shape[0] == 0:
        return

    if coords.shape[0] > MAX_RESIDUES:
        return

    state=contact_mapgen(coords, pdb_id, tensor_dir)

    if state:
        FASTA_seqgen(sorted_nodes, pdb_id, fasta_dir)
        print(f"Built {pdb_id}")


def build_split(split):

    split_dir=os.path.join(DATA_ROOT, split)

    csv_path=os.path.join(split_dir, f"{split}.csv")
    fasta_dir=os.path.join(split_dir, "FASTA")
    tensor_dir=os.path.join(split_dir, "Tensors")

    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(tensor_dir, exist_ok=True)

    df=pd.read_csv(csv_path)

    if "PDB" not in df.columns:
        df["PDB"]=df["ID"].astype(str).str[:4].str.lower()

    pdb_ids=df["PDB"].unique()

    for pdb_id in pdb_ids:
        build_protein(pdb_id, fasta_dir, tensor_dir)


if __name__ == "__main__":

    for split in ["train", "val", "test"]:
        build_split(split)
